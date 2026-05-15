import pymilvus
import random, datetime
import numpy as np
from mpmath import limit
from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)

# ==================== 配置部分 ====================
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "example_books"
DIMENSION = 128  # 向量维度

# ==================== 1. 连接Milvus ====================
def connect_milvus() -> MilvusClient:
    """连接到Milvus服务"""
    print(f"正在连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    client = MilvusClient(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        db_name="hello_milvus"
    )

    print("✓ 连接成功！")
    print(f"Milvus Server版本: {client.get_server_version()}, pymilvus版本: {pymilvus.__version__}")
    print("-" * 50)
    return client

# ==================== 2. 创建集合（表） ====================
def create_collection(client: MilvusClient):
    """创建集合（表）"""
    if client.has_collection(COLLECTION_NAME):
        print(f"集合 '{COLLECTION_NAME}' 已存在，正在删除...")
        client.drop_collection(COLLECTION_NAME)
        print(f"✓ 旧集合已删除")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="书籍主键ID"
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=DIMENSION,
            description="文本向量"
        ),
        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=200,
            description="书籍名称"
        ),
        FieldSchema(
            name="author",
            dtype=DataType.VARCHAR,
            max_length=100,
            description="作者"
        ),
        FieldSchema(
            name="category",
            dtype=DataType.VARCHAR,
            max_length=50,
            description="分类"
        ),
        FieldSchema(
            name="price",
            dtype=DataType.FLOAT,
            description="价格"
        ),
        FieldSchema(
            name="publish_year",
            dtype=DataType.INT64,
            description="出版年份"
        ),
        FieldSchema(
            name="created_at",
            dtype=DataType.VARCHAR,
            max_length=50,
            description="创建时间"
        )
    ]

    schema = CollectionSchema(fields=fields, description="书籍信息集合", enable_dynamic_field=True)
    collection = client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    print(f"✓ 集合 '{COLLECTION_NAME}' 创建成功")
    print(f"  字段列表: {[f.name for f in fields]}")
    print("-" * 50)
    return collection

# ==================== 3. 生成测试数据 ====================
def generate_test_data(num_records=100):
    """生成测试数据"""
    # 书籍数据样本
    books_data = [
        # 书名, 作者, 分类, 价格, 年份
        ("三体", "刘慈欣", "科幻", 68.0, 2008),
        ("三体II：黑暗森林", "刘慈欣", "科幻", 78.0, 2008),
        ("三体III：死神永生", "刘慈欣", "科幻", 88.0, 2010),
        ("流浪地球", "刘慈欣", "科幻", 45.0, 2008),
        ("球状闪电", "刘慈欣", "科幻", 52.0, 2005),
        ("活着", "余华", "文学", 45.0, 1993),
        ("许三观卖血记", "余华", "文学", 48.0, 1995),
        ("兄弟", "余华", "文学", 68.0, 2005),
        ("第七天", "余华", "文学", 39.0, 2013),
        ("文城", "余华", "文学", 59.0, 2021),
        ("平凡的世界", "路遥", "文学", 128.0, 1986),
        ("人生", "路遥", "文学", 39.0, 1982),
        ("白鹿原", "陈忠实", "文学", 69.0, 1993),
        ("百年孤独", "加西亚·马尔克斯", "外国文学", 55.0, 1967),
        ("霍乱时期的爱情", "加西亚·马尔克斯", "外国文学", 52.0, 1985),
        ("挪威的森林", "村上春树", "外国文学", 45.0, 1987),
        ("海边的卡夫卡", "村上春树", "外国文学", 58.0, 2002),
        ("人类简史", "尤瓦尔·赫拉利", "历史", 68.0, 2011),
        ("未来简史", "尤瓦尔·赫拉利", "历史", 68.0, 2015),
        ("今日简史", "尤瓦尔·赫拉利", "历史", 58.0, 2018),
    ]

    # 如果请求的数据量大于样本数，循环使用样本并添加随机数据
    data_list = []
    for i in range(num_records):
        # 循环使用书籍样本
        book = books_data[i % len(books_data)]

        # 生成随机向量（模拟文本的语义向量）
        # 在实际应用中，这里应该是通过embedding模型生成的向量
        embedding = np.random.randn(DIMENSION).astype(np.float32)
        # 归一化（可选，提高检索效果）
        embedding = embedding / np.linalg.norm(embedding)

        # 为重复的书名添加序号
        title = book[0]
        if i >= len(books_data):
            title = f"{book[0]} (版本{i // len(books_data) + 1})"

        data = {
            # "id": i + 1,  # ID从1开始
            "embedding": embedding.tolist(),
            "title": title,
            "author": book[1],
            "category": book[2],
            "price": book[3] + random.uniform(-10, 10),  # 价格微调
            "publish_year": book[4],
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        data_list.append(data)

    print(f"✓ 生成了 {len(data_list)} 条测试数据")
    print(f"  数据示例: {data_list[0]}")
    print("-" * 50)
    return data_list

# ==================== 4. 插入数据 (Create) ====================
def insert_data(client: MilvusClient, data_list):
    """插入数据到集合"""
    # 插入数据
    print(f"正在插入 {len(data_list)} 条数据...")
    insert_result = client.insert(
        collection_name=COLLECTION_NAME,
        data=data_list,
        timeout=60
    )

    print(f"✓ 插入成功！")
    print(f"  插入ID: {insert_result['ids'][:5]}... (共{len(insert_result['ids'])}条)")
    print(f"  插入时间戳: {insert_result}")
    # 刷新数据，确保数据可被搜索
    client.flush(collection_name=COLLECTION_NAME)
    print("✓ 数据已刷新到磁盘")
    print("-" * 50)
    return insert_result

# ==================== 5. 创建索引 (为搜索做准备) ====================
def create_index(client: MilvusClient):
    """创建向量索引，加速搜索"""

    # 索引参数配置
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",  # 索引类型
        metric_type="COSINE", # 相似度度量方式: COSINE, L2, IP
        params = {"nlist": 128}  # 索引参数
    )

    print("正在创建向量索引...")
    client.create_index(index_params=index_params, collection_name=COLLECTION_NAME)

    print(f"✓ 索引创建成功")
    print(f"  索引类型: {index_params[0]['index_type']}")
    print(f"  度量方式: {index_params[0]['metric_type']}")
    print("-" * 50)

# ==================== 6. 查询数据 (Read) ====================
def search_similar(client: MilvusClient, query_vector, top_k=5):
    """向量相似度搜索"""

    # 加载 Collection
    client.load_collection(COLLECTION_NAME)
    print("✓ 集合已加载")

    # 搜索参数
    search_params = {
        "metric_type": "COSINE",
        "params": {
            "nprobe": 10
        }
    }
    print(f"\n正在搜索与查询向量最相似的 {top_k} 条数据...")

    # 执行搜索
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        search_params=search_params,
        limit=top_k,
        output_fields=[
            "id",
            "title",
            "author",
            "category",
            "price",
            "publish_year"
        ]
    )

    # 打印结果
    print("\n搜索结果:")
    print("-" * 80)

    for hits in results:
        for hit in hits:
            print(hit)
    print("-" * 50)
    return results


def query_by_ids(client: MilvusClient, entity_ids:list):
    """根据ID查询实体"""

    print(f"\n查询 ID={entity_ids} 的数据...")

    # 使用ID查询
    result = client.query(
        collection_name=COLLECTION_NAME,
        ids=entity_ids,
        output_fields=["id", "title", "author", "category", "price", "publish_year", "created_at"])
    if result:
        print(f"✓ 找到数据:")
        for key, value in result[0].items():
            print(f"  {key}: {value}")
    else:
        print(f"✗ 未找到 ID={entity_ids} 的数据")

    print("-" * 50)
    return result


def query_by_condition(client: MilvusClient, category, min_price, max_price):
    """条件查询：按分类和价格范围筛选"""

    print(f"\n查询条件: category='{category}', 价格在 [{min_price}, {max_price}] 之间")

    # 构建查询表达式
    expr = f"category == '{category}' and price >= {min_price} and price <= {max_price}"

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter=expr,
        output_fields=["id", "title", "author", "price", "publish_year"],
        limit=10
    )

    print(f"✓ 找到 {len(result)} 条数据:")
    for item in result:
        print(f"  [{item['id']}] {item['title']} - {item['author']} - ¥{item['price']:.2f} - {item['publish_year']}")

    print("-" * 50)
    return result


# ==================== 7. 更新数据 (Update) ====================
def update_entity(client: MilvusClient, expr:str, entity_id, **kwargs):
    """更新实体数据（通过先删除后插入的方式）"""

    print(f"\n正在更新 ID={entity_id} 的数据...")

    # 先查询原数据
    old_data = client.query(
        collection_name=COLLECTION_NAME,
        ids=[entity_id],
        filter=expr,
        output_fields=["id", "title", "author", "embedding", "category", "price", "publish_year", "created_at"],
        limit=1
    )

    if not old_data:
        print(f"✗ 未找到 ID={entity_id} or {expr} 的数据，无法更新")
        return None

    print(f"\n正在更新 ID={entity_id} or {expr} 的数据...")

    new_data = old_data[0]
    for key, value in kwargs.items():
        if value is not None:
            new_data[key] = value
    # 删除旧数据
    client.delete(collection_name=COLLECTION_NAME, ids=[entity_id], filter=expr)
    print(f"  - 已删除旧数据 (ID: {entity_id})")

    # 新增
    upsert_result = client.upsert(collection_name=COLLECTION_NAME, data=[new_data])
    client.flush(collection_name=COLLECTION_NAME)
    print(f"  - 已插入新数据: {upsert_result}")
    print(f"✓ 更新成功！")
    print("-" * 50)

    return new_data


# ==================== 8. 删除数据 (Delete) ====================
def delete_data(client: MilvusClient, entity_ids:list):
    """删除指定ID的数据"""

    print(f"\n正在删除 ID={entity_ids} 的数据...")

    # 先检查是否存在
    check_result = client.query(
        collection_name=COLLECTION_NAME,
        ids=entity_ids,
        output_fields=["id", "title"]
    )

    if not check_result:
        print(f"✗ 未找到 ID={entity_ids} 的数据")
        return False

    # 执行删除
    client.delete(collection_name=COLLECTION_NAME, ids=entity_ids)
    client.flush(collection_name=COLLECTION_NAME)

    print(f"✓ 成功删除数据: {check_result[0]}")
    print("-" * 50)
    return True


# ==================== 9. 统计信息 ====================
def get_collection_stats(client: MilvusClient):
    """获取集合统计信息"""

    print("\n=== 集合统计信息 ===")

    # collection 基本信息
    collection_info = client.describe_collection(
        collection_name=COLLECTION_NAME
    )

    print(f"集合名称: {COLLECTION_NAME}")

    # 数据量
    stats = client.get_collection_stats(
        collection_name=COLLECTION_NAME
    )

    print(f"数据条数: {stats['row_count']}")

    # schema 信息
    fields = collection_info["fields"]

    print(f"字段数量: {len(fields)}")

    print(
        f"字段列表: {[field['name'] for field in fields]}"
    )

    # 索引信息
    indexes = client.list_indexes(
        collection_name=COLLECTION_NAME
    )

    if indexes:

        print(f"索引列表: {indexes}")

        for index_name in indexes:

            index_info = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=index_name
            )

            print(f"\n索引名称: {index_name}")
            print(f"索引信息: {index_info}")

    print("-" * 50)


# ==================== 10. 清理资源 ====================
def cleanup(client: MilvusClient):
    """清理资源"""

    print("\n正在释放资源...")

    # 从内存卸载 collection
    client.release_collection(
        collection_name=COLLECTION_NAME
    )

    print("✓ 集合已从内存释放")

    # 可选：删除整个 collection
    # client.drop_collection(COLLECTION_NAME)
    # print(f"✓ 集合 '{COLLECTION_NAME}' 已删除")

    # 关闭连接
    client.close()

    print("✓ 已断开 Milvus 连接")

if __name__ == '__main__':
    the_client = connect_milvus()

    # book_collection = create_collection(the_client)
    #
    # books = generate_test_data(num_records=100)
    #
    # insert_data(the_client, books)

    # create_index(the_client)

    # vector = np.random.randn(DIMENSION).astype(np.float32)
    # top5_results = search_similar(the_client, vector, 5)

    # 466314611599094362
    # query_by_ids(the_client, [466314611599094362])
    # query_by_condition(the_client, "历史", 0, 100)
    # update_entity(the_client, expr=None, entity_id=466314611599094423, price=10.0, title="三体II：黑暗森林 v9")
    # delete_data(the_client, entity_ids=[466314611599094323])
    # get_collection_stats(the_client)
    cleanup(the_client)