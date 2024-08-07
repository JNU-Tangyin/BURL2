# BURL2
Behavior-aware URL embedding with Lookup Table  
1.data_preprocess：  
    p = Preprocess()  
    p.fenxi_host_split()  # 生成复用对齐域名列表并保存在当前目录下  
    graph = p.build_graph()  # 生成知识图谱字典并保存在当前目录下  
2.viusal_graoh:
    运行，在浏览器查看可视化的图  
    数据集要求有以下字段：['标签', 'len'，'0', '1', '2', '3', '4', '5', '6', '7']  
                   解释: 标签为对应url的分类，len为域名去掉http://后，.split('.')的长度，数字字段为对应.split('.')取倒序后的各个项。（默认域名长度最长为7，这里即各级域名）
    构建图会去掉可能出现的顶级域名（例如com, cn, top等)
