# Recent progress in semantic image segmentation

Xiaolong Liu, Zhidong Deng, Yuhan Yang (2018)

## 🧩 Problem to Solve

본 논문은 이미지 처리 및 컴퓨터 비전 분야의 핵심 응용 기술인 시맨틱 이미지 분할(Semantic Image Segmentation)의 발전 과정을 분석하는 것을 목표로 한다. 시맨틱 분할은 이미지의 모든 픽셀을 해당 객체의 클래스로 분류하는 '픽셀 수준의 분류(pixel-level classification)' 작업이다.

이 문제는 의료 영상 분석(뇌 및 종양 탐지, 수술 도구 추적)과 지능형 교통 시스템(자율 주행 자동차의 도로 표지판 인식 및 장면 해석)과 같은 다양한 도메인에서 매우 중요하다. 특히 자율 주행 시스템의 경우, 주변 환경을 정확히 이해하기 위한 장면 해석(Scene Parsing) 능력이 필수적이며, 이는 시맨틱 분할 기술에 크게 의존한다. 따라서 본 논문은 전통적인 방식부터 최신 딥러닝(DNN) 기반 방식까지의 흐름을 정리하여 연구자들에게 가이드를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 주된 기여는 방대한 양의 시맨틱 분할 연구를 체계적으로 분류하고 분석한 리뷰 보고서를 제공하는 것이다. 특히 DNN 기반의 최신 방법론을 다음과 같은 8가지 핵심 관점으로 세분화하여 분석하였다.

1. **Fully Convolutional Network (FCN)**: 전결합층을 컨볼루션 층으로 대체하는 구조적 혁신.
2. **Up-sample 방법**: 보간법(Interpolation)과 디컨볼루션(Deconvolution)의 비교.
3. **FCN과 CRF의 결합**: 정교한 경계 복원을 위한 Conditional Random Field(CRF) 및 Domain Transform(DT) 적용.
4. **Dilated Convolution**: 해상도 손실 없이 수용 영역(Receptive Field)을 확장하는 기법.
5. **Backbone Network의 발전**: VGG, ResNet, ResNeXt, Inception 등 기본 네트워크의 진화.
6. **Pyramid 방법**: Image Pyramid, ASPP(Atrous Spatial Pyramid Pooling), PSPNet의 Pyramid Pooling 등 다중 스케일 정보 추출 전략.
7. **Multi-level 및 Multi-stage 방법**: Hypercolumns 및 Deep Layer Cascade(LC)를 통한 정밀도 향상.
8. **학습 체계**: Supervised, Weakly-supervised, Unsupervised 학습 방법론의 검토.

## 📎 Related Works

### 전통적인 방법 (Traditional Methods)

DNN 등장 이전에는 수동으로 설계된 특징점(Hand-crafted features)과 분류기에 의존하였다.

- **특징 추출**: HOG, SIFT, LBP, SURF, Harris Corners 등 다양한 특징 추출기가 사용되었다.
- **분류 및 분할**: 임계값 설정(Thresholding), K-means 클러스터링, 에너지 모델, Edge-based 탐지, Support Vector Machine(SVM) 등이 활용되었다.
- **그래프 모델**: Markov Random Network(MRF)와 Conditional Random Field(CRF)가 픽셀 간의 관계를 모델링하여 최적의 레이블을 찾는 데 사용되었다.
- **한계**: 수동 특징 설계의 한계로 인해 복잡한 장면에서의 일반화 성능이 낮았다.

### 데이터셋 및 평가 지표

논문은 연구자들이 알고리즘을 검증하기 위해 사용하는 주요 데이터셋과 지표를 소개한다.

- **데이터셋**: PASCAL VOC, MS COCO, ADE20K, Cityscapes, KITTI 등이 있으며, 특히 Cityscapes와 KITTI는 자율 주행 환경에 특화되어 있다.
- **평가 지표**: 픽셀 정확도($P_{acc}$), 평균 정확도($M_{acc}$), 평균 교집합-합집합 비율($M_{IU}$), 빈도 가중 IU($FW_{IU}$) 등을 사용한다.

## 🛠️ Methodology

본 논문은 특정 알고리즘을 제안하는 것이 아니라 기존 방법론들을 분석하고 설명한다. 핵심적인 기술적 구성 요소는 다음과 같다.

### 1. 평가 지표의 수학적 정의

분할 성능을 측정하기 위해 다음과 같은 수식을 사용한다. 여기서 $n_{ij}$는 클래스 $i$의 픽셀을 클래스 $j$로 잘못 예측한 수이며, $t_i$는 클래스 $i$의 전체 픽셀 수이다.

- **Pixel Accuracy**: $\displaystyle P_{acc} = \frac{\sum_i n_{ii}}{\sum_i t_i}$
- **Mean Accuracy**: $\displaystyle M_{acc} = \frac{1}{n_{cl}} \sum_i \frac{n_{ii}}{t_i}$
- **Mean Intersection over Union (mIoU)**: $\displaystyle M_{IU} = \frac{1}{n_{cl}} \sum_i \frac{n_{ii}}{t_i + \sum_j n_{ji} - n_{ii}}$
- **Frequency Weighted IU**: $\displaystyle FW_{IU} = \frac{1}{\sum_k t_k} \sum_i \frac{t_i n_{ii}}{t_i + \sum_j n_{ji} - n_{ii}}$

### 2. DNN 기반 분할 파이프라인의 핵심 요소

- **FCN (Fully Convolutional Network)**: 기존 분류 네트워크의 Fully Connected layer를 Convolutional layer로 대체하여 입력 이미지의 크기에 관계없이 픽셀 단위의 맵을 출력한다. 해상도 복원을 위해 보간법(Interpolation)을 사용하며, 낮은 수준의 특징과 높은 수준의 특징을 결합하는 Skip connection을 도입한다.
- **Up-sampling**:
  - **Interpolation**: 단순한 선형 보간법으로 계산 효율이 높다.
  - **Deconvolution**: 학습 가능한 파라미터를 통해 특징 맵의 크기를 복원하는 역-컨볼루션 과정이다.
- **Dilated Convolution (Atrous Convolution)**: 커널의 필터 사이에 간격(hole)을 두어 파라미터 수를 늘리지 않고도 수용 영역(Receptive Field)을 기하급수적으로 확장한다. 이는 해상도 저하 없이 다중 스케일 문맥 정보를 통합하는 데 유리하다.
- **Pyramid Strategies**:
  - **ASPP (Atrous Spatial Pyramid Pooling)**: 서로 다른 샘플링 비율을 가진 여러 Dilated Convolution 필터를 병렬로 적용하여 다양한 크기의 객체를 동시에 포착한다.
  - **Pyramid Pooling Module (PSPNet)**: 특징 맵을 서로 다른 크기의 영역으로 풀링한 후, 이를 다시 업샘플링하여 결합함으로써 전역적인 문맥 정보(Global Context)를 확보한다.

## 📊 Results

본 논문은 서베이 논문이므로 여러 기존 연구의 정량적 결과들을 인용하여 설명한다.

- **FCN**: PASCAL VOC 데이터셋에서 기존 방식 대비 상대적으로 20%의 성능 향상을 보이며 mIoU 62.2%를 달성하였다.
- **DeepLab (FCN + CRF)**: DNN의 마지막 층에 CRF를 결합하여 픽셀 위치의 불확실성을 해결하였으며, PASCAL VOC-2012 테스트 세트에서 71.6%의 IOU 정확도를 기록하였다.
- **PSPNet**: Pyramid Pooling Module을 통해 PASCAL VOC 2012에서 85.4% mIoU, Cityscapes에서 80.2% mIoU라는 기록적인 성과를 거두었다.
- **Backbone의 영향**: VGG-16에서 ResNet, ResNeXt 등으로 백본 네트워크가 심화됨에 따라 시맨틱 분할의 정확도 또한 지속적으로 향상되는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문을 통해 도출할 수 있는 분석적 통찰은 다음과 같다.

첫째, 시맨틱 분할의 핵심 난제는 **'해상도(Spatial Resolution)'와 '의미론적 정보(Semantic Information)' 사이의 트레이드-오프**를 해결하는 것이다. 딥러닝 네트워크가 깊어질수록 강력한 의미 정보는 얻을 수 있지만, 풀링 과정을 통해 공간 해상도가 손실된다. 이를 해결하기 위해 Skip connection, Dilated Convolution, 그리고 다양한 Pyramid 구조가 제안되었음을 알 수 있다.

둘째, 백본 네트워크(Backbone Network)의 발전이 분할 성능에 직접적인 영향을 미친다. 이미지 분류 작업에서 검증된 ResNet이나 Inception 같은 구조가 분할 작업의 feature extractor로서 필수적인 역할을 수행하며, 네트워크 아키텍처의 진화가 분할 성능의 상한선을 높이고 있다.

셋째, 단순한 딥러닝 모델을 넘어 CRF나 Domain Transform 같은 전통적인 최적화 방법론을 결합하거나, Multi-stage cascade 방식을 통해 쉬운 영역과 어려운 영역을 구분하여 처리하는 전략이 효율성을 높이는 방향으로 발전하고 있다.

## 📌 TL;DR

본 논문은 전통적인 특징 기반 방식에서 최신 딥러닝 기반 방식으로 진화한 시맨틱 이미지 분할 기술을 종합적으로 분석한 리뷰 논문이다. 특히 FCN을 기점으로 Dilated Convolution, ASPP, PSPNet과 같은 구조적 혁신이 어떻게 픽셀 수준의 분류 정확도를 높였는지 체계적으로 정리하였다. 이 연구는 시맨틱 분할의 전체적인 기술 지형도를 제공함으로써, 향후 다중 스케일 문맥 정보 활용 및 효율적인 업샘플링 연구의 기초 자료로 활용될 가치가 크다.
