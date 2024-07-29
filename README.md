# BURL2
Behavior-aware URL embedding with Lookup Table  
1.data_preprocess：  
    p = Preprocess()  
    p.fenxi_host_split()  # 生成复用对齐域名列表并保存在当前目录下  
    graph = p.build_graph()  # 生成知识图谱字典并保存在当前目录下  
