import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import re
from random import shuffle


class Preprocess:
    def __init__(self, **config):
        self.clean_count_len = None
        self.max_len = config.get("max_len", 7)
        # 在我们的数据集中url域名级别最低为6(顶级，1级，2级，3级，4级，5级，6级）
        self.data = self.load_data(config.get("data_path"))
        self.host_column = config.get("host_column_name", "主机地址")
        self.label_column = config.get("label_column_name", "标签")

    def load_data(self, path):
        try:
            data = pd.read_excel(path).fillna("")
        except:
            data = pd.read_csv(path).fillna("")
        return data

    def clean(self):
        # 去掉复用地址和ip
        print("去除复用地址和ip")

        data = self.data.drop_duplicates(subset=['主机地址', '标签'])
        data = data.loc[self.data['主机地址'].apply(self.drop_ip), :]
        data = data.drop_duplicates(subset=['主机地址'], keep=False)
        return data

    def drop_ip(self, x):
        # 去掉ip地址，字符串中去掉.和:，若能进行类型转换，则为纯数字，为ip地址
        try:
            int(x.replace('.', '').replace(':', ''))
            return False
        except:
            return True

    def count_duplicate(self):
        # 统计每个主机地址出现的次数
        s = self.data.groupby('主机地址').count()['标签']
        df = self.clean()
        for i in df.index:
            df.loc[i, 'num'] = s[df.loc[i, '主机地址']]
        return df

    def make_new_dataset(self, path='clean_count.csv', save_path='clean_ds.csv'):
        print("创建供机器学习的数据集，对小类别按比例扩充")
        data = pd.read_csv(path)
        df_li = []
        total = data.loc[:, 'num'].sum()

        with tqdm(total=total) as pbar:
            for i in data.index:
                repeat_num = data.loc[i, 'num']
                for j in range(int(repeat_num)):
                    df_li.append(data.loc[i, :].tolist())
                    pbar.update(1)
            df = pd.DataFrame(df_li, columns=data.columns)
            df.to_csv(save_path, index=False)

        # 纯按重复次数清洗，某些不必要的url会重复很多次，比如一个分类下的一个url，重复五百次，毫无意义。

    def count_clean_count_len(self):
        df = self.count_duplicate()
        for i in df.index:
            df.loc[i, "len"] = df.loc[i, "主机地址"].split('.').__len__()
        self.clean_count_len = df
        return df

    def host_split(self):
        # host_split.csv
        print("按域名级别对齐")
        df = self.count_clean_count_len()
        max_len = df.loc[:, 'len'].max()
        self.max_len = max_len
        host_s = pd.DataFrame(columns=df.columns.append(pd.Index(range(int(max_len)))))
        with tqdm(total=df.loc[:, 'num'].count()) as pbar:
            for i in df.index:
                pbar.update(1)
                host_s.loc[i, df.columns] = df.loc[i, :]
                if ':' in df.loc[i, '主机地址']:
                    host = re.findall(r'(.*):', df.loc[i, '主机地址'])[0]
                else:
                    host = df.loc[i, '主机地址']
                # 分割并倒序
                data = host.split('.')
                data.reverse()

                for col, val in enumerate(data):
                    host_s.loc[i, col] = val

        return host_s

    def fenxi_host_split(self):
        df = self.host_split()

        # print(df.loc[:, '0'].unique())    #提取顶级域名位，交给wenxin chatbot得到顶级域名和非顶级域名的列表，写道filtter.py中
        def drop_duplicates_on_domain(top_i: int = 0):
            use_df: pd.DataFrame = df.copy(deep=True).loc[:, ['序号', '标签', top_i]]
            use_df = use_df.dropna()
            print("去空值，即去掉不存在{}级域名的地址{}个".format(top_i, df.__len__() - use_df.__len__()))
            # 先按标签和domain去重
            dropped = use_df.drop_duplicates(['标签', str(top_i)])
            duplicates = dropped.loc[dropped.duplicated([str(top_i)]), :]
            print("{}级域名中的复用域名有{}个".format(top_i, duplicates.__len__()))
            return duplicates.loc[:, str(top_i)].unique().tolist()

        duplicate_li = []
        print("提取对齐后的复用域名, 存为duplicate.thg, duplicate.thg: 列表[[],[]...]索引是n级域名，值是复用列表")
        for i in range(int(self.max_len)):
            duplicate_li.append(drop_duplicates_on_domain(i))

        with open("duplicate.thg", "wb") as w:
            pickle.dump(duplicate_li, w)

    def build_graph(self):
        print("创建图")
        if self.clean_count_len is None:
            self.count_clean_count_len()
        try:
            # 创建过滤器
            f = Filter()
        except:
            # 无复用域名表，则创建复用域名表后创建过滤器
            self.fenxi_host_split()
            f = Filter()
            ...
        for i in self.clean_count_len.index:
            self.clean_count_len.loc[i, '识别符'] = str(f(self.clean_count_len.loc[i, '主机地址']))

        graph = {}
        l = self.clean_count_len.index.tolist()
        shuffle(l)
        for i in l:
            if str(self.clean_count_len.loc[i, '识别符']) == '{}':
                ...
            else:
                graph[self.clean_count_len.loc[i, '识别符']] = self.clean_count_len.loc[i, '标签']

        with open("graph.thg", "wb") as w:
            pickle.dump(graph, w)

        return graph


class Filter:
    def __init__(self):
        with open("duplicate.thg", "rb") as r:
            self.duplicate_li = pickle.load(r)

    def __call__(self, host: str):
        assert '/' not in host
        if ':' in host:
            host = re.findall(r'(.*):', host)[0]
        else:
            host = host
        host_li = host.split('.')
        host_li.reverse()
        res = {}
        for loc, i in enumerate(host_li):
            if i not in self.duplicate_li[loc]:
                res[loc] = i
        return res


if __name__ == '__main__':
    # use example:
    p = Preprocess()
    p.fenxi_host_split()
    graph = p.build_graph()
