-- 数据库 Schema 设计

-- 机构表
CREATE TABLE IF NOT EXISTS institutions (
    institution_id INTEGER PRIMARY KEY AUTOINCREMENT, -- 机构ID，自增
    institution_name TEXT NOT NULL, -- 机构名称，string，不可为空
    country_region TEXT NOT NULL, -- 国家/地区，string，不可为空
    region TEXT, -- 区域分类（亚洲、北美、欧洲等），string，可为空
    is_chinese_mainland INTEGER DEFAULT 0, -- 是否为中国大陆，bool，默认False
    UNIQUE(institution_name, country_region) -- 唯一键，机构名称和国家/地区
);

-- 学科表
CREATE TABLE IF NOT EXISTS research_fields (
    field_id INTEGER PRIMARY KEY AUTOINCREMENT, -- 学科ID，自增
    field_name TEXT NOT NULL UNIQUE -- 学科名称，string，不可为空，唯一
);

-- 排名数据表
CREATE TABLE IF NOT EXISTS ranking_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- 排名数据ID，自增
    field_id INTEGER, -- 学科ID，int
    institution_id INTEGER, -- 机构ID，int
    rank_position INTEGER, -- 排名位置，int
    wos_documents INTEGER, -- Web of Science Documents，int
    cites INTEGER, -- Cites，int
    cites_per_paper REAL, -- Cites/Paper，real，2位小数
    top_papers INTEGER, -- Top Papers，int
    FOREIGN KEY (field_id) REFERENCES research_fields(field_id), -- 外键，学科ID
    FOREIGN KEY (institution_id) REFERENCES institutions(institution_id) -- 外键，机构ID
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_field_rank ON ranking_data(field_id, rank_position); -- 索引，学科ID和排名位置
CREATE INDEX IF NOT EXISTS idx_institution ON ranking_data(institution_id); -- 索引，机构ID

