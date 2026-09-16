// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package parquet

import (
	"context"
	"os"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/apache/arrow/go/v17/parquet"
	"github.com/apache/arrow/go/v17/parquet/pqarrow"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/v3/objectstorage"
)

type listRow[T any] struct {
	data  []T
	valid bool
}

func collectListLike[T any](t *testing.T, listReader *listLikeArray, getElement func(int) (T, error)) []listRow[T] {
	t.Helper()
	var rows []listRow[T]
	err := getListLikeArrayData(listReader, getElement, func(arr []T, valid bool) {
		rows = append(rows, listRow[T]{data: arr, valid: valid})
	})
	require.NoError(t, err)
	return rows
}

func int64ListField(nullable bool) *schemapb.FieldSchema {
	return &schemapb.FieldSchema{
		FieldID:     100,
		Name:        "arr",
		DataType:    schemapb.DataType_Array,
		ElementType: schemapb.DataType_Int64,
		Nullable:    nullable,
		TypeParams: []*commonpb.KeyValuePair{
			{Key: "max_capacity", Value: "50"},
		},
	}
}

func boolListField(nullable bool) *schemapb.FieldSchema {
	return &schemapb.FieldSchema{
		FieldID:     100,
		Name:        "arr",
		DataType:    schemapb.DataType_Array,
		ElementType: schemapb.DataType_Bool,
		Nullable:    nullable,
		TypeParams: []*commonpb.KeyValuePair{
			{Key: "max_capacity", Value: "50"},
		},
	}
}

func stringListField(nullable bool) *schemapb.FieldSchema {
	return &schemapb.FieldSchema{
		FieldID:     100,
		Name:        "arr",
		DataType:    schemapb.DataType_Array,
		ElementType: schemapb.DataType_VarChar,
		Nullable:    nullable,
		TypeParams: []*commonpb.KeyValuePair{
			{Key: "max_capacity", Value: "50"},
			{Key: "max_length", Value: "256"},
		},
	}
}

func collectionSchema(field *schemapb.FieldSchema) *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{field}}
}

func buildInt64ListArray(t *testing.T, rows [][]int64, valid []bool, elemNullAt []int) arrow.Array {
	t.Helper()
	builder := array.NewListBuilder(memory.DefaultAllocator, arrow.PrimitiveTypes.Int64)
	t.Cleanup(builder.Release)
	vb := builder.ValueBuilder().(*array.Int64Builder)
	elemNull := map[int]struct{}{}
	for _, idx := range elemNullAt {
		elemNull[idx] = struct{}{}
	}
	elemIdx := 0
	for i, row := range rows {
		if valid != nil && !valid[i] {
			builder.AppendNull()
			continue
		}
		builder.Append(true)
		for _, v := range row {
			if _, isNull := elemNull[elemIdx]; isNull {
				vb.AppendNull()
			} else {
				vb.Append(v)
			}
			elemIdx++
		}
	}
	arr := builder.NewArray()
	t.Cleanup(arr.Release)
	return arr
}

func buildBoolListArray(t *testing.T, rows [][]bool, valid []bool) arrow.Array {
	t.Helper()
	builder := array.NewListBuilder(memory.DefaultAllocator, arrow.FixedWidthTypes.Boolean)
	t.Cleanup(builder.Release)
	vb := builder.ValueBuilder().(*array.BooleanBuilder)
	for i, row := range rows {
		if valid != nil && !valid[i] {
			builder.AppendNull()
			continue
		}
		builder.Append(true)
		vb.AppendValues(row, nil)
	}
	arr := builder.NewArray()
	t.Cleanup(arr.Release)
	return arr
}

func buildStringListArray(t *testing.T, rows [][]string, valid []bool) arrow.Array {
	t.Helper()
	builder := array.NewListBuilder(memory.DefaultAllocator, arrow.BinaryTypes.String)
	t.Cleanup(builder.Release)
	vb := builder.ValueBuilder().(*array.StringBuilder)
	for i, row := range rows {
		if valid != nil && !valid[i] {
			builder.AppendNull()
			continue
		}
		builder.Append(true)
		vb.AppendValues(row, nil)
	}
	arr := builder.NewArray()
	t.Cleanup(arr.Release)
	return arr
}

func writeArrayParquet(t *testing.T, schema *schemapb.CollectionSchema, arr arrow.Array, maxRowGroup int64) string {
	t.Helper()
	file, err := os.CreateTemp(t.TempDir(), "empty_array_*.parquet")
	require.NoError(t, err)
	defer file.Close()

	arrowSchema := arrow.NewSchema([]arrow.Field{{
		Name:     schema.Fields[0].GetName(),
		Type:     arr.DataType(),
		Nullable: schema.Fields[0].GetNullable(),
	}}, nil)

	writer, err := pqarrow.NewFileWriter(
		arrowSchema,
		file,
		parquet.NewWriterProperties(parquet.WithMaxRowGroupLength(maxRowGroup)),
		pqarrow.DefaultWriterProps(),
	)
	require.NoError(t, err)

	record := array.NewRecord(arrowSchema, []arrow.Array{arr}, int64(arr.Len()))
	defer record.Release()
	require.NoError(t, writer.Write(record))
	require.NoError(t, writer.Close())
	return file.Name()
}

func readArrayParquet(t *testing.T, schema *schemapb.CollectionSchema, path string) *storage.InsertData {
	t.Helper()
	ctx := context.Background()
	factory := storage.NewChunkManagerFactory("local", objectstorage.RootPath(testOutputPath))
	cm, err := factory.NewPersistentStorageChunkManager(ctx)
	require.NoError(t, err)
	reader, err := NewReader(ctx, cm, schema, path, 64*1024*1024)
	require.NoError(t, err)
	t.Cleanup(reader.Close)
	insertData, err := reader.Read()
	require.NoError(t, err)
	return insertData
}

func TestGetListLikeArrayDataEmptyVsNull(t *testing.T) {
	field := int64ListField(true)
	arr := buildInt64ListArray(t, [][]int64{
		{1, 2},
		{},
		nil,
		{3},
		{},
	}, []bool{true, true, false, true, true}, nil)
	listReader, err := newListLikeArray(arr, field)
	require.NoError(t, err)

	int64Reader := listReader.ListValues().(*array.Int64)
	rows := collectListLike(t, listReader, func(i int) (int64, error) {
		if int64Reader.IsNull(i) {
			return 0, WrapNullElementErr(field)
		}
		return int64Reader.Value(i), nil
	})

	require.Equal(t, []listRow[int64]{
		{data: []int64{1, 2}, valid: true},
		{data: []int64{}, valid: true},
		{data: nil, valid: false},
		{data: []int64{3}, valid: true},
		{data: []int64{}, valid: true},
	}, rows)
}

func TestGetListLikeArrayDataAllEmpty(t *testing.T) {
	field := int64ListField(true)
	arr := buildInt64ListArray(t, [][]int64{{}, {}, {}}, []bool{true, true, true}, nil)
	listReader, err := newListLikeArray(arr, field)
	require.NoError(t, err)
	int64Reader := listReader.ListValues().(*array.Int64)
	rows := collectListLike(t, listReader, func(i int) (int64, error) {
		return int64Reader.Value(i), nil
	})
	require.Equal(t, []listRow[int64]{
		{data: []int64{}, valid: true},
		{data: []int64{}, valid: true},
		{data: []int64{}, valid: true},
	}, rows)
}

func TestGetListLikeArrayDataAllNull(t *testing.T) {
	field := int64ListField(true)
	arr := buildInt64ListArray(t, [][]int64{nil, nil, nil}, []bool{false, false, false}, nil)
	listReader, err := newListLikeArray(arr, field)
	require.NoError(t, err)
	rows := collectListLike(t, listReader, func(i int) (int64, error) {
		t.Fatalf("null rows must not read elements")
		return 0, nil
	})
	require.Equal(t, []listRow[int64]{
		{data: nil, valid: false},
		{data: nil, valid: false},
		{data: nil, valid: false},
	}, rows)
}

func TestGetListLikeArrayDataSliced(t *testing.T) {
	field := int64ListField(true)
	arr := buildInt64ListArray(t, [][]int64{
		{1},
		{},
		nil,
		{2, 3},
		{4},
	}, []bool{true, true, false, true, true}, nil)
	sliced := array.NewSlice(arr, 1, 4)
	t.Cleanup(sliced.Release)

	listReader, err := newListLikeArray(sliced, field)
	require.NoError(t, err)
	int64Reader := listReader.ListValues().(*array.Int64)
	rows := collectListLike(t, listReader, func(i int) (int64, error) {
		if int64Reader.IsNull(i) {
			return 0, WrapNullElementErr(field)
		}
		return int64Reader.Value(i), nil
	})
	require.Equal(t, []listRow[int64]{
		{data: []int64{}, valid: true},
		{data: nil, valid: false},
		{data: []int64{2, 3}, valid: true},
	}, rows)
}

func TestGetListLikeArrayDataBoolAndVarChar(t *testing.T) {
	t.Run("bool", func(t *testing.T) {
		field := boolListField(true)
		arr := buildBoolListArray(t, [][]bool{{true}, {}, nil, {false, true}}, []bool{true, true, false, true})
		listReader, err := newListLikeArray(arr, field)
		require.NoError(t, err)
		var rows []listRow[bool]
		err = readBoolListLikeData(field, listReader, func(data []bool, valid bool) {
			rows = append(rows, listRow[bool]{data: data, valid: valid})
		})
		require.NoError(t, err)
		require.Equal(t, []listRow[bool]{
			{data: []bool{true}, valid: true},
			{data: []bool{}, valid: true},
			{data: nil, valid: false},
			{data: []bool{false, true}, valid: true},
		}, rows)
	})

	t.Run("varchar", func(t *testing.T) {
		field := stringListField(true)
		arr := buildStringListArray(t, [][]string{{"a"}, {}, nil, {"b", "c"}}, []bool{true, true, false, true})
		listReader, err := newListLikeArray(arr, field)
		require.NoError(t, err)
		var rows []listRow[string]
		err = readStringListLikeData(field, listReader, func(string) error { return nil }, func(data []string, valid bool) {
			rows = append(rows, listRow[string]{data: data, valid: valid})
		})
		require.NoError(t, err)
		require.Equal(t, []listRow[string]{
			{data: []string{"a"}, valid: true},
			{data: []string{}, valid: true},
			{data: nil, valid: false},
			{data: []string{"b", "c"}, valid: true},
		}, rows)
	})
}

func TestGetListLikeArrayDataNullElement(t *testing.T) {
	field := int64ListField(true)
	arr := buildInt64ListArray(t, [][]int64{{1, 2}}, []bool{true}, []int{1})
	listReader, err := newListLikeArray(arr, field)
	require.NoError(t, err)
	err = readIntegerOrFloatListLikeData[int64](field, listReader, func([]int64, bool) {})
	require.Error(t, err)
	require.ErrorContains(t, err, "array element is not allowed to be null value")
}

func TestGetListLikeArrayDataFixedSizeListNull(t *testing.T) {
	field := int64ListField(true)
	builder := array.NewFixedSizeListBuilder(memory.DefaultAllocator, 2, arrow.PrimitiveTypes.Int64)
	vb := builder.ValueBuilder().(*array.Int64Builder)
	builder.Append(true)
	vb.AppendValues([]int64{1, 2}, nil)
	builder.AppendNull()
	builder.Append(true)
	vb.AppendValues([]int64{3, 4}, nil)
	arr := builder.NewArray()
	t.Cleanup(func() {
		arr.Release()
		builder.Release()
	})

	listReader, err := newListLikeArray(arr, field)
	require.NoError(t, err)
	int64Reader := listReader.ListValues().(*array.Int64)
	rows := collectListLike(t, listReader, func(i int) (int64, error) {
		if int64Reader.IsNull(i) {
			return 0, WrapNullElementErr(field)
		}
		return int64Reader.Value(i), nil
	})
	require.Equal(t, []listRow[int64]{
		{data: []int64{1, 2}, valid: true},
		{data: nil, valid: false},
		{data: []int64{3, 4}, valid: true},
	}, rows)
}

func TestParquetImportEmptyArrayNotNull(t *testing.T) {
	field := int64ListField(true)
	schema := collectionSchema(field)
	arr := buildInt64ListArray(t, [][]int64{
		{1, 2},
		{},
		nil,
		{3},
		{},
	}, []bool{true, true, false, true, true}, nil)
	path := writeArrayParquet(t, schema, arr, int64(arr.Len()))
	insertData := readArrayParquet(t, schema, path)

	got := insertData.Data[field.FieldID]
	require.Equal(t, 5, got.RowNum())
	require.Equal(t, []int64{1, 2}, got.GetRow(0).(*schemapb.ScalarField).GetLongData().GetData())
	require.Empty(t, got.GetRow(1).(*schemapb.ScalarField).GetLongData().GetData())
	require.Nil(t, got.GetRow(2))
	require.Equal(t, []int64{3}, got.GetRow(3).(*schemapb.ScalarField).GetLongData().GetData())
	require.Empty(t, got.GetRow(4).(*schemapb.ScalarField).GetLongData().GetData())
}

func TestParquetImportEmptyArrayAllEmptyAndAllNull(t *testing.T) {
	t.Run("all_empty", func(t *testing.T) {
		field := int64ListField(true)
		schema := collectionSchema(field)
		arr := buildInt64ListArray(t, [][]int64{{}, {}, {}}, []bool{true, true, true}, nil)
		path := writeArrayParquet(t, schema, arr, 1)
		insertData := readArrayParquet(t, schema, path)
		got := insertData.Data[field.FieldID]
		require.Equal(t, 3, got.RowNum())
		for i := 0; i < 3; i++ {
			require.NotNil(t, got.GetRow(i))
			require.Empty(t, got.GetRow(i).(*schemapb.ScalarField).GetLongData().GetData())
		}
	})

	t.Run("all_null", func(t *testing.T) {
		field := int64ListField(true)
		schema := collectionSchema(field)
		arr := buildInt64ListArray(t, [][]int64{nil, nil, nil}, []bool{false, false, false}, nil)
		path := writeArrayParquet(t, schema, arr, 1)
		insertData := readArrayParquet(t, schema, path)
		got := insertData.Data[field.FieldID]
		require.Equal(t, 3, got.RowNum())
		for i := 0; i < 3; i++ {
			require.Nil(t, got.GetRow(i))
		}
	})
}

func TestParquetImportEmptyArrayBoolAndVarChar(t *testing.T) {
	t.Run("bool", func(t *testing.T) {
		field := boolListField(true)
		schema := collectionSchema(field)
		arr := buildBoolListArray(t, [][]bool{{true, false}, {}, nil}, []bool{true, true, false})
		path := writeArrayParquet(t, schema, arr, int64(arr.Len()))
		insertData := readArrayParquet(t, schema, path)
		got := insertData.Data[field.FieldID]
		require.Equal(t, []bool{true, false}, got.GetRow(0).(*schemapb.ScalarField).GetBoolData().GetData())
		require.Empty(t, got.GetRow(1).(*schemapb.ScalarField).GetBoolData().GetData())
		require.Nil(t, got.GetRow(2))
	})

	t.Run("varchar", func(t *testing.T) {
		field := stringListField(true)
		schema := collectionSchema(field)
		arr := buildStringListArray(t, [][]string{{"x"}, {}, nil}, []bool{true, true, false})
		path := writeArrayParquet(t, schema, arr, int64(arr.Len()))
		insertData := readArrayParquet(t, schema, path)
		got := insertData.Data[field.FieldID]
		require.Equal(t, []string{"x"}, got.GetRow(0).(*schemapb.ScalarField).GetStringData().GetData())
		require.Empty(t, got.GetRow(1).(*schemapb.ScalarField).GetStringData().GetData())
		require.Nil(t, got.GetRow(2))
	})
}

func TestParquetImportNonNullableEmptyArray(t *testing.T) {
	field := int64ListField(false)
	schema := collectionSchema(field)
	arr := buildInt64ListArray(t, [][]int64{{1}, {}, {2, 3}}, nil, nil)
	path := writeArrayParquet(t, schema, arr, int64(arr.Len()))
	insertData := readArrayParquet(t, schema, path)
	got := insertData.Data[field.FieldID]
	require.Equal(t, 3, got.RowNum())
	require.Equal(t, []int64{1}, got.GetRow(0).(*schemapb.ScalarField).GetLongData().GetData())
	require.Empty(t, got.GetRow(1).(*schemapb.ScalarField).GetLongData().GetData())
	require.Equal(t, []int64{2, 3}, got.GetRow(2).(*schemapb.ScalarField).GetLongData().GetData())
}

func TestParquetImportNullElementStillRejected(t *testing.T) {
	field := int64ListField(true)
	schema := collectionSchema(field)
	arr := buildInt64ListArray(t, [][]int64{{1, 2}}, []bool{true}, []int{1})
	path := writeArrayParquet(t, schema, arr, 1)

	ctx := context.Background()
	factory := storage.NewChunkManagerFactory("local", objectstorage.RootPath(testOutputPath))
	cm, err := factory.NewPersistentStorageChunkManager(ctx)
	require.NoError(t, err)
	reader, err := NewReader(ctx, cm, schema, path, 64*1024*1024)
	require.NoError(t, err)
	defer reader.Close()
	_, err = reader.Read()
	require.Error(t, err)
	require.ErrorContains(t, err, "array element is not allowed to be null value")
}
