# `TensorFlow PG`

## `preparation`

- tensorflow==1.12
- seaborn
- numpy
- pandas
- matplotlib
- jupyter notebook
- python==3.6

## `tensorflow`实战

- 房价预测模型
- `MINIST`识别
- 验证码识别
- 人脸识别

### 验证码识别

#### 1. 搭建环境 

`pip install Pillow captcha pydot flask -i http://pypi.douban.com/simple --trusted-host pypi.douban.com`

- 数据集成: `Pillow captcha`
- 模型可视化: `pydot`
- 模型服务部署: `flask`

#### 2. 验证码生成

- 验证码：一种区分用户是计算机或人的公共全自动程序
- `CAPTCHA`: 反向图灵测试

使用 `Pillow` 和 `captcha` 生成

```python
import PIL
import captcha

fp = "./data/a.jpg"
PIL.Image.Open(fp, mode="r")  # 打开识别的文件

captcha.image.ImageCaptcha(width, height,) # 创建 ImageCaptcha 实例
captcha.image.ImageCaptcha.write('1234', 'out.png') # 生成验证码并保存
captcha.image.ImageCaptcha.generate('1234') # 生成验证码图像
```

#### 3. 输入数据处理

图像处理: RGB => 灰度图 => 规范化数据

## `Tensorflow 2.0`起步

### 环境配置

- `NAVIDIA GTX965`
- `CUDA==10.0`
- `cudnn==7.6`
- `tensorflow==2.0.0-beta1`


 