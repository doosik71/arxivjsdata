# Recent progress in semantic image segmentation

Xiaolong Liu, Zhidong Deng, Yuhan Yang (2018)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 및 이미지 처리 분야의 핵심 과제인 Semantic Image Segmentation(의미론적 이미지 분할)의 발전 과정을 체계적으로 분석하고 정리하는 것을 목표로 한다. Semantic Image Segmentation은 이미지의 각 픽셀을 미리 정의된 클래스 중 하나로 분류하는 '픽셀 수준의 분류(pixel-level classification)' 작업이다.

이 문제는 단순히 객체의 위치를 찾는 Object Detection이나 이미지 전체의 카테고리를 결정하는 Image Classification보다 훨씬 정밀한 예측을 요구한다. 특히 의료 영상 분석(종양 검출 등)이나 자율주행 시스템의 도로 장면 분석(Scene Parsing)과 같은 고정밀 응용 분야에서 그 중요성이 매우 높다. 따라서 본 논문은 전통적인 방식부터 최신 Deep Neural Network(DNN) 기반 방식까지의 기술적 흐름을 정리하여 연구자들에게 가이드를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 주된 기여는 파편화되어 있는 Semantic Image Segmentation의 방법론들을 체계적인 분류 체계에 따라 통합적으로 분석했다는 점이다. 저자들은 세그멘테이션 방법론을 크게 '전통적 방법(Traditional methods)'과 '최근의 DNN 방법(Recent DNN methods)'으로 구분하며, 특히 DNN 기반 방법론을 다음의 8가지 핵심 관점에서 상세히 분석한다.

1. **Fully Convolutional Network (FCN)**: 구조적 전환점 분석
2. **Up-sample 방식**: Interpolation과 Deconvolution의 비교
3. **FCN과 CRF의 결합**: 후처리 과정을 통한 정밀도 향상
4. **Dilated Convolution**: 해상도 손실 없는 수용 영역(Receptive Field) 확장
5. **Backbone Network의 발전**: VGG에서 ResNet, ResNeXt 등으로의 진화
6. **Pyramid 방법론**: 다중 스케일 특징 추출 전략
7. **Multi-level Feature 및 Multi-stage 방법**: 정밀한 위치 정보와 의미 정보의 결합
8. **학습 감독 수준**: Supervised, Weakly-supervised, Unsupervised 방법론

## 📎 Related Works

논문에서는 DNN 등장 이전의 전통적인 접근 방식들을 다룬다. 과거에는 픽셀 색상, HOG(Histogram of Oriented Gradients), SIFT(Scale-Invariant Feature Transform), LBP(Local Binary Pattern)와 같은 수동 설계된 특징(Hand-crafted features)을 추출하고, 이를 SVM(Support Vector Machine)이나 MRF(Markov Random Field), CRF(Conditional Random Field)와 같은 분류기 및 그래프 모델에 입력하는 방식이 주를 이뤘다.

전통적 방법의 한계는 복잡한 장면에서 고차원적인 의미 정보를 학습하는 능력이 부족하다는 점이며, 이를 해결하기 위해 DNN, 특히 Convolutional Neural Network(CNN)가 도입되면서 성능이 비약적으로 향상되었다.

## 🛠️ Methodology

본 논문은 리뷰 논문으로서 특정 알고리즘을 제안하기보다, 기존의 핵심 방법론들을 기술적으로 설명한다.

### 1. 성능 평가 지표 (Evaluation Metrics)
세그멘테이션 성능을 측정하기 위해 다음과 같은 네 가지 지표를 정의한다. 여기서 $n_{ij}$는 클래스 $i$의 픽셀을 클래스 $j$로 잘못 예측한 수, $t_i$는 클래스 $i$의 실제 픽셀 수, $n_{cl}$은 전체 클래스 수를 의미한다.

- **Pixel Accuracy ($P_{acc}$)**: 전체 픽셀 중 정확히 예측된 픽셀의 비율이다.
$$P_{acc} = \frac{\sum_{i} n_{ii}}{\sum_{i} t_i}$$

- **Mean Accuracy ($M_{acc}$)**: 각 클래스별 정확도의 평균이다.
$$M_{acc} = \frac{1}{n_{cl}} \sum_{i} \frac{n_{ii}}{t_i}$$

- **Mean Intersection over Union ($M_{IU}$)**: 예측 영역과 실제 영역의 합집합 대비 교집합의 비율의 평균이다.
$$M_{IU} = \frac{1}{n_{cl}} \sum_{i} \frac{n_{ii}}{t_i + \sum_{j} n_{ji} - n_{ii}}$$

- **Frequency Weighted IU ($FW_{IU}$)**: 클래스별 픽셀 빈도를 가중치로 둔 IU이다.
$$FW_{IU} = \frac{1}{\sum_{k} t_k} \sum_{i} \frac{t_i n_{ii}}{t_i + \sum_{j} n_{ji} - n_{ii}}$$

### 2. 핵심 DNN 아키텍처 및 기법

**Fully Convolutional Network (FCN)**
FCN의 핵심 아이디어는 기존 CNN의 Fully Connected(FC) 레이어를 Convolution 레이어로 대체하는 것이다. 이를 통해 임의의 크기의 입력 이미지를 처리할 수 있으며, Interpolation 레이어를 통해 출력 크기를 입력 크기와 동일하게 복원하여 픽셀 단위의 예측을 수행한다.

**Up-sampling 방식**
- **Bilinear Interpolation**: 계산 효율성이 높으며 원래 이미지의 크기를 복구하는 데 널리 사용된다.
- **Deconvolution**: Convolution의 역연산으로, 학습 가능한 파라미터를 통해 특징 맵의 크기를 복구한다.

**Dilated Convolution (Atrous Convolution)**
필터 요소 사이에 간격을 두어 수용 영역(Receptive Field)을 확장하는 기법이다. 이는 풀링(Pooling) 연산으로 인해 발생하는 해상도 저하 없이도 넓은 문맥(Context) 정보를 수집할 수 있게 한다.

**Pyramid 방법론**
- **Image Pyramid**: 입력 이미지를 다양한 크기로 리사이징하여 입력하는 방식이다.
- **ASPP (Atrous Spatial Pyramid Pooling)**: 서로 다른 샘플링 비율을 가진 Dilated Convolution 필터들을 병렬로 배치하여 다중 스케일의 특징을 동시에 캡처한다.
- **PSPNet (Pyramid Scene Parsing Network)**: 다양한 크기의 Pooling 영역을 통해 전역 문맥 정보를 집계하여 정밀도를 높인다.

**Backbone Network의 진화**
초기에는 VGG-16, AlexNet 등이 사용되었으나, 이후 잔차 학습(Residual Learning)을 도입한 ResNet과 이를 확장한 ResNeXt, Inception-v4 등으로 발전하며 더 깊은 층에서도 안정적인 학습이 가능해졌다.

## 📊 Results

본 논문은 개별 실험 결과보다는 주요 벤치마크 데이터셋과 기존 모델들의 성과를 정리하여 제시한다.

- **사용된 데이터셋**:
    - **PASCAL VOC**: 20개 클래스를 포함하는 가장 일반적인 벤치마크이다.
    - **MS COCO**: 91개 객체 타입을 포함하며 대규모 라벨링 데이터를 제공한다.
    - **ADE20K**: 150개의 객체 및 배경 클래스를 포함하며, 부분(part) 세그멘테이션 마스크를 제공한다.
    - **Cityscapes**: 자율주행을 위한 도시 거리 장면 데이터셋으로 30개 클래스를 포함한다.
    - **KITTI**: 자율주행 관련 도로 검출 및 3D 객체 탐지 데이터셋이다.

- **주요 성과**:
    - FCN은 PASCAL VOC 2012에서 기존 방식 대비 약 20%의 상대적 성능 향상을 보였다.
    - PSPNet은 Pyramid Pooling Module을 통해 PASCAL VOC 2012에서 $mIoU$ 85.4%, Cityscapes에서 80.2%라는 기록적인 성과를 달성하였다.
    - DeepLab 시리즈는 Dense CRF를 결합하여 PASCAL VOC-2012 테스트 세트에서 71.6%의 $IoU$ 정확도를 달성하였다.

## 🧠 Insights & Discussion

본 논문을 통해 분석한 Semantic Image Segmentation의 핵심 통찰은 **'해상도 유지'와 '문맥 파악' 사이의 트레이드-오프(Trade-off) 해결**이다. 

전통적인 CNN은 풀링 연산을 통해 수용 영역을 넓히지만, 이 과정에서 공간 해상도가 손실되어 경계선 예측이 부정확해지는 문제가 발생한다. 이를 해결하기 위해 FCN의 Skip connection, DeepLab의 Dilated Convolution, 그리고 PSPNet의 Pyramid Pooling과 같은 기법들이 등장하였다. 특히, 얕은 층의 정밀한 위치 정보(Localization)와 깊은 층의 풍부한 의미 정보(Semantics)를 어떻게 효과적으로 결합하느냐가 성능 향상의 핵심이다.

또한, Fully Supervised 학습의 높은 비용 문제를 해결하기 위해 Weakly-supervised나 Unsupervised 방법론에 대한 연구가 진행되고 있다는 점은 향후 데이터 효율성 측면에서 중요한 연구 방향이 될 것으로 보인다.

## 📌 TL;DR

본 논문은 전통적인 특징 추출 방식에서 최신 DNN 기반의 Semantic Image Segmentation까지의 발전 과정을 종합적으로 리뷰한 보고서이다. 특히 **FCN $\rightarrow$ Dilated Convolution $\rightarrow$ Pyramid Pooling $\rightarrow$ Advanced Backbones**로 이어지는 기술적 흐름을 체계화하였으며, 이는 자율주행 및 의료 영상 분석과 같이 픽셀 단위의 정밀한 이해가 필요한 분야의 연구자들에게 필수적인 기술적 지도를 제공한다.