# 2020年服务外包比赛A10浪潮无监督分类系统

## A10浪潮无监督分类系统介绍

**业务背景** 

“一贷通”是公司“一平七通”发展战略中的“一通”之一，“一贷通”的 业务目标旨在积极采用大数据、机器学习、人工智能等先进的金融科技技术，汇 聚各个政府委办局、区域内金融机构、互联网等多渠道的基础数据，搭建一涵盖 各金融业务数据的多功能的数字金融科技服务平台，形成科学、客观、可靠的中 小微企业信用评分体系，着力解决中小企业“融资难、融资贵”的问题。企业多 源数据、多维度的深入挖掘是为企业构建企业画像、建立企业信用评分体系的前 提基础，从企业的企业背景、经营能力、经营风险、发展状况等层面对企业进行 群体划分，企业划分结果中的每一个企业簇群体都要形成较明显的标签标示，为 后续企业画像构建、企业信用评分体系构建提供辅助。

**问题说明** 

以某一地市的小微企业为研究对象，以该地市小微企业覆盖企业背景、企业 稳定性、企业经营能力、企业经营风险、司法风险、信用风险等多个方面的数据 作为数据来源。建立一种无监督的分类模型，利用小微企业包含的特征维度信息， 对小微企业进行簇划分，划分的每一个簇都有有效的特征或者标签去描述该簇的 特征，每个簇之间形成较为明显的划分界限，即最终形成企业合理的划分。

**用户期望**

 追求企业无标识脱敏数据的有效划分及每个簇划分标签的合理有效且可区 分:

1.针对无标识的企业数据进行数据预处理，特征備选，特征提取等形成 有效的训练样例及特征；

 2.针对提取的有效特征选择合适的无监督分类方法对小微企业数据进行 分类，进行模型训练，模型要求实现小微企业群体的有效划分；

 3.针对小微企业划分后各簇提取显著标签进行该簇的描述，要求标签合 理且有效；

**团队分工及安排：**

- 团队：白日依山尽

- 成员

|ID|Work|
|:-:|:-:|
|Peony|后端设计|
|喋喋不休|算法设计|
|MOSS-A-134|前端设计|
|LMLF|UI设计|
|七七|文档编写|

## Todolist

- **后端**

- [ ] 网站搭建
- [ ] 路径编写
- [ ] 接口调试

- **UI设计**

- **前端**

- **产品设计**

- [x] 登录板块功能设计
    - [x] 登录页面
    - [x] 注册页面
    - [x] 找回页面
- [x] 搜索板块功能设计
    - [x] 简洁式搜索页面（仿搜索引擎）
    - [ ] 服务推送式搜索页面
- [ ] 展示板块功能设计

- **文档编写**

- **算法**

- [ ] 数据清洗
    - [x] 数据读取
    - [ ] 数据属性判断
    - [x] 无量纲化处理
        - [x] 定性数据转换
            - [x] One-Hot encoding
            - [x] Label encoding
        - [x] 缩放处理
            - [x] 标准化处理
            - [x] 最值缩放处理
    - [ ] 缺失值处理
        - [ ] 缺失值占比大
        - [ ] 缺失值占比小
- [ ] 特征选择
    - [ ] 算法选择
    - [ ] 冗余特征的剔除
- [ ] 聚类分析
- [ ] 结果分析