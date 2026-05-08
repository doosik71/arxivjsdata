# CE-Net: Context Encoder Network for 2D Medical Image Segmentation

Zaiwang Gu, Jun Cheng, Huazhu Fu, Kang Zhou, Huaying Hao, Yitian Zhao, Tianyang Zhang, Shenghua Gao and Jiang Liu (2019)

## 🧩 Problem to Solve

본 논문은 2D 의료 영상 분할(Medical Image Segmentation)에서 발생하는 정보 손실 문제를 해결하고자 한다. 의료 영상 분할은 질병 진단 및 분석의 핵심적인 단계이며, 최근 딥러닝, 특히 U-Net 기반의 구조가 널리 사용되고 있다. 하지만 U-Net과 그 변형 모델들은 연속적인 Pooling 연산과 Strided Convolution을 통해 특징 맵의 해상도를 낮추어 추상적인 특징을 학습하는데, 이 과정에서 세밀한 공간 정보(Spatial Information)가 손실되는 문제가 발생한다.

이러한 공간 정보의 손실은 정밀한 경계 예측이 필요한 Dense Prediction 작업, 즉 분할 작업의 성능을 저해하는 요인이 된다. 따라서 본 연구의 목표는 고수준의 시맨틱 정보(High-level semantic information)를 충분히 캡처하면서도, 분할 성능 향상을 위해 공간 정보를 최대한 보존할 수 있는 새로운 네트워크 구조인 CE-Net을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Encoder-Decoder 구조 사이에 **Context Extractor** 모듈을 삽입하여, 다양한 크기의 객체에 대응할 수 있는 다중 스케일 컨텍스트 정보를 추출하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Dense Atrous Convolution (DAC) 블록 제안**: Atrous Convolution(Dilated Convolution)을 계층적으로 배치하여, 파라미터 수를 크게 늘리지 않고도 수용 영역(Receptive Field)을 확장함으로써 더 넓고 깊은 시맨틱 특징을 추출한다.
2. **Residual Multi-kernel Pooling (RMP) 블록 제안**: 다양한 크기의 Pooling 커널을 사용하여 전역 컨텍스트 정보를 인코딩하며, 이를 통해 의료 영상 내 객체 크기의 심한 변동성에 대응한다.
3. **범용적 프레임워크 구축**: 제안한 모듈들을 ResNet-34 기반의 Encoder-Decoder 구조에 통합하여, 시신경 유두 분할, 혈관 검출, 폐 분할, 세포 윤곽 분할, 망막 OCT 층 분할 등 서로 다른 5가지 의료 영상 작업에서 SOTA(State-of-the-art) 성능을 달성함을 입증하였다.

## 📎 Related Works

의료 영상 분할의 기존 접근 방식은 크게 세 가지 단계로 구분된다.

- **전통적 방법**: Edge Detection, Template Matching, Hough Transform 등을 사용하였으나, 수작업으로 설계된 특징(Hand-crafted features)에 의존하므로 일반화 능력이 부족하고 설계가 어렵다는 한계가 있다.
- **초기 딥러닝 방법**: Image Patch 기반의 CNN이나 Sliding Window 방식을 사용하였으나, 중복 계산이 많고 전역 특징(Global features)을 학습하지 못하는 단점이 있다.
- **U-Net 및 그 변형**: Fully Convolutional Network(FCN)의 발전으로 등장한 U-Net은 Encoder-Decoder 구조를 통해 의료 영상 분할의 표준이 되었다. 이후 CRF(Conditional Random Field) 결합, Deep Supervision 추가, Residual Connection 도입 등의 변형이 시도되었다.

본 논문은 U-Net 계열이 공통적으로 가진 '해상도 감소로 인한 공간 정보 손실'이라는 한계를 지적하며, 이를 해결하기 위해 Atrous Convolution과 Multi-scale Pooling을 결합한 Context Extractor를 도입함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

CE-Net은 크게 **Feature Encoder**, **Context Extractor**, **Feature Decoder**의 세 가지 모듈로 구성된다.

### 1. Feature Encoder Module

기존 U-Net의 단순한 Convolution 블록 대신 ImageNet으로 사전 학습된(Pre-trained) **ResNet-34**의 처음 4개 블록을 사용한다. ResNet의 Shortcut mechanism은 그래디언트 소실(Gradient vanishing) 문제를 방지하고 네트워크의 수렴 속도를 가속화한다.

### 2. Context Extractor Module

본 논문의 핵심 모듈로, DAC 블록과 RMP 블록으로 이루어져 있다.

- **Dense Atrous Convolution (DAC)**:
  Atrous Convolution은 필터 요소 사이에 0을 삽입하여 수용 영역을 넓히는 연산이다. 수학적으로 2D 신호에 대한 연산은 다음과 같다.
  $$y[i] = \sum_{k} x[i+rk]w[k]$$
  여기서 $r$은 Atrous rate를 의미한다. DAC 블록은 4개의 캐스케이드 브랜치를 가지며, Atrous rate를 1, 1, 3, 5로 점진적으로 증가시켜 수용 영역을 3, 7, 9, 19로 확장한다. 이를 통해 다양한 크기의 객체 특징을 동시에 추출하며, 마지막에 Shortcut connection을 통해 입력 특징을 더해준다.

- **Residual Multi-kernel Pooling (RMP)**:
  객체 크기의 변화에 대응하기 위해 $2\times2, 3\times3, 5\times5, 6\times6$의 4가지 서로 다른 크기의 Pooling 커널을 사용한다.
  1. 각 커널로 Pooling을 수행한 후, $1\times1$ Convolution을 통해 채널 차원을 $\frac{1}{N}$으로 축소하여 계산 비용을 줄인다.
  2. Bilinear Interpolation을 통해 원래 특징 맵 크기로 업샘플링한다.
  3. 최종적으로 원본 특징 맵과 업샘플링된 특징 맵들을 Concatenation 한다.

### 3. Feature Decoder Module

인코더에서 추출된 고수준 특징을 복원한다. 정보 손실을 보완하기 위해 Encoder의 특징 맵을 직접 전달하는 Skip Connection을 사용한다. 특히 단순한 업샘플링 대신 **Transposed Convolution**(Deconvolution)을 사용하여 학습 가능한 매핑을 통해 더 정밀한 세부 정보를 복원한다. 디코더 블록은 $1\times1$ Conv $\rightarrow$ $3\times3$ Transposed Conv $\rightarrow$ $1\times1$ Conv 순서로 구성된다.

### 4. Loss Function

의료 영상의 특성상 배경 대비 객체가 차지하는 영역이 매우 작기 때문에, 일반적인 Cross Entropy 대신 **Dice coefficient loss**를 사용한다.
$$L_{dice} = 1 - \frac{K \sum_{k} 2\omega_k \sum_{i} p(k,i)g(k,i)}{\sum_{i} p^2(k,i) + \sum_{i} g^2(k,i)}$$
여기서 $p(k,i)$는 예측 확률, $g(k,i)$는 Ground Truth 라벨이다. 최종 손실 함수는 Overfitting을 방지하기 위한 정규화 항($L_{reg}$)을 포함하여 다음과 같이 정의된다.
$$L_{loss} = L_{dice} + L_{reg}$$

## 📊 Results

본 연구는 5가지 서로 다른 의료 영상 분할 작업에서 성능을 검증하였다.

| 작업 (Task) | 데이터셋 | 주요 지표 | 결과 및 분석 |
| :--- | :--- | :--- | :--- |
| **시신경 유두 분할** | ORIGA, Messidor, RIM-ONE-R1 | Overlapping Error ($E$) | ORIGA에서 $E=0.058$ 달성, 기존 SOTA 대비 약 15.9% 성능 향상 |
| **망막 혈관 검출** | DRIVE | Sen, Acc, AUC | Sen(0.8309), Acc(0.9545), AUC(0.9779)로 기존 방법들보다 우수 |
| **폐 분할** | LUNA | $E$, Acc, Sen | Overlapping Error를 0.038까지 낮추어 U-Net 및 Backbone 대비 성능 향상 |
| **세포 윤곽 분할** | EM Challenge | $V_{Rand}, V_{Info}$ | $V_{Rand}(0.9743), V_{Info}(0.9878)$로 가장 높은 점수 기록 |
| **망막 OCT 층 분할** | Topcon | Mean Absolute Error (MAE) | 전체 MAE 1.68 달성, U-Net(2.45) 대비 31.4% 오차 감소 |

**Ablation Study 결과**:

- **Pre-trained ResNet**: 사전 학습된 가중치를 사용했을 때 학습 손실이 훨씬 빠르게 감소하며 성능이 향상됨을 확인하였다.
- **DAC vs Regular Conv**: Atrous Convolution을 사용한 DAC가 일반 Convolution보다 고수준 시맨틱 특징 추출에 훨씬 유리함을 입증하였다.
- **RMP의 효과**: RMP 모듈을 추가했을 때 시신경 유두 분할의 Overlapping Error가 유의미하게 감소하였다.
- **복잡도 비교**: 유사한 네트워크 복잡도를 가진 Inception-ResNet-V2 기반 모델보다 CE-Net의 성능이 더 우수함을 확인하여, 성능 향상이 단순한 파라미터 증가 때문이 아님을 증명하였다.

## 🧠 Insights & Discussion

본 논문은 제안한 CE-Net이 단일 클래스 분할뿐만 아니라 망막 OCT 층 분할과 같은 **다중 클래스(Multi-class) 분할 작업**에서도 효과적임을 보여주었다. 이는 DAC와 RMP 모듈이 제공하는 다중 스케일 컨텍스트 정보가 다양한 형태와 크기의 의료 영상 객체를 구분하는 데 매우 강력한 도구가 됨을 시사한다.

또한, 사전 학습된 ResNet을 백본으로 사용하고 Dice Loss를 적용함으로써 데이터셋이 부족한 의료 영상 환경에서도 안정적인 학습과 높은 일반화 성능을 확보하였다.

다만, 본 연구는 2D 영상 분할에 집중하고 있으며, 실제 의료 현장에서 많이 사용되는 3D 볼륨 데이터(CT, MRI 등)로의 확장성에 대해서는 명시적으로 다루지 않았다. 향후 연구에서 3D 데이터로의 확장이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 U-Net의 고질적인 문제인 공간 정보 손실을 해결하기 위해, 다중 스케일 특징을 효율적으로 추출하는 **Dense Atrous Convolution (DAC)**과 **Residual Multi-kernel Pooling (RMP)** 모듈을 도입한 **CE-Net**을 제안한다. 이 모델은 5가지의 서로 다른 2D 의료 영상 분할 작업에서 기존 SOTA 모델들을 뛰어넘는 성능을 보였으며, 특히 다양한 크기의 객체를 정밀하게 분할하는 데 탁월한 능력을 입증하였다. 이는 향후 다양한 2D 의료 영상 분석 시스템의 백본 구조로 활용될 가능성이 높다.
