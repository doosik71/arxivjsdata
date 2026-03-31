# Refine Myself by Teaching Myself : Feature Refinement via Self-Knowledge Distillation

Mingi Ji, Seungjae Shin, Seunghyun Hwang, Gibeom Park, Il-Chul Moon (2021)

## 🧩 Problem to Solve

해결하고자 하는 문제는 기존 Knowledge Distillation (KD) 및 Self-knowledge Distillation (SKD) 방법론의 한계를 극복하는 것이다.

- **해결하고자 하는 문제**:
    1. **기존 Knowledge Distillation (KD)의 한계**: KD는 대규모의 사전 학습된 Teacher 모델로부터 Student 모델로 지식을 전달하여 모델 압축 및 성능 향상을 도모한다. 그러나 이 방식은 복잡하고 큰 Teacher 모델을 사전 학습하는 데 막대한 컴퓨팅 자원을 요구하며, Teacher 네트워크의 선택에 따라 Student 네트워크의 성능이 가변적이라는 문제점을 지닌다.
    2. **기존 Self-knowledge Distillation (SKD)의 한계**: SKD는 사전 학습된 Teacher 모델 없이 네트워크 스스로 지식을 증류하여 학습을 강화하는 방법이다.
        - **데이터 증강 기반 SKD**: 데이터 증강을 통해 모델의 일관된 예측을 유도하지만, 증강 과정에서 이미지의 로컬(local) 정보(공간적 정보)가 손실될 수 있다. 이는 Semantic Segmentation과 같이 로컬 정보가 중요한 비전 태스크에 적용하기 어렵게 만들며, Feature-map Distillation 적용 또한 어렵게 한다.
        - **보조 네트워크 기반 SKD**: 분류기 네트워크 중간에 보조 브랜치를 활용하여 지식을 전달하지만, 보조 네트워크가 분류기 네트워크와 동일하거나 낮은 복잡도를 가지므로 "정제된 지식(refined knowledge)" (특히 feature map 형태의 지식)을 효과적으로 생성하고 전달하는 데 한계가 있다.

- **문제의 중요성**: 모바일 기기와 같은 제한된 컴퓨팅 자원 환경에서 딥러닝 모델을 효율적으로 배포하고 높은 성능을 유지하는 것은 중요한 과제이다. SKD는 이러한 요구사항을 충족할 잠재력이 있지만, 기존 방법론들의 한계로 인해 다양한 비전 태스크에 대한 일반화된 적용이 어려웠다. 따라서, 로컬 정보 보존이 중요하고 정제된 지식을 활용할 수 있는 새로운 SKD 방법론의 개발은 모델의 효율성과 적용 범위를 넓히는 데 필수적이다.

- **논문의 목표**: 기존 SKD 방법론의 한계를 극복하고, 정제된 Feature-map과 Soft Label을 활용하여 Student 네트워크(Classifier Network)의 성능을 효과적으로 향상시키는 새로운 Self-knowledge Distillation 방법인 Feature Refinement via Self-Knowledge Distillation (FRSKD)를 제안하는 것이다. FRSKD는 특히 로컬 정보 보존이 중요한 분류 및 Semantic Segmentation과 같은 다양한 비전 태스크에 적용 가능하며, 다른 SKD 방법 및 데이터 증강 기법과 호환성을 가진다.

## ✨ Key Contributions

본 논문의 핵심적인 직관과 설계 아이디어는 다음과 같다.

- **Self-teacher 네트워크를 통한 정제된 지식 생성**: 기존 Self-knowledge Distillation (SKD) 방법들이 보조 네트워크를 사용하더라도 정제된 지식(refined knowledge)을 효과적으로 생성하지 못하거나, 데이터 증강 과정에서 로컬 정보를 손실하는 한계가 있었다. 이를 해결하기 위해, 본 논문은 분류기 네트워크에 정제된 feature-map과 soft label을 제공하는 **보조 self-teacher 네트워크**를 도입한다. 이 self-teacher 네트워크는 메인 분류기 네트워크가 스스로를 지도하여 학습 성능을 향상시키는 역할을 수행한다.

- **BiFPN 기반의 효율적인 Self-teacher 구조**: 제안하는 self-teacher 네트워크는 객체 탐지 분야에서 multi-scale feature를 효과적으로 처리하는 것으로 알려진 BiFPN 구조를 분류 태스크에 맞게 변형하여 설계되었다. Top-down 및 Bottom-up 경로를 통해 다양한 스케일의 feature-map을 집계하고 정제함으로써, 기존 SKD 방식이 제공하기 어려웠던 "정제된 지식"을 스스로 생성할 수 있다.

- **채널 차원 조절을 통한 효율성 극대화**: Self-teacher 네트워크의 lateral convolutional layer에서 출력 채널 차원 $d_i$를 해당 feature map의 입력 채널 차원 $c_i$에 비례하도록 ($d_i = w \times c_i$) 조절하는 방식을 제안한다. 이는 각 레이어의 깊이에 따른 정보량을 반영하면서도, 네트워크 파라미터와 FLOPs를 크게 줄여 전체 시스템의 계산 효율성을 높인다. 이를 통해 복잡한 feature 네트워크를 경량화하면서도 효과적인 feature-map 정제 및 지식 전달을 가능하게 한다.

- **Soft Label 및 Feature-map Distillation의 통합**: FRSKD는 self-teacher 네트워크가 생성한 soft label을 통한 지식 증류($L_{KD}$)뿐만 아니라, 정제된 feature-map을 활용하는 feature distillation($L_F$, 특히 attention transfer 기반)을 함께 사용한다. 이는 특히 로컬 정보 보존이 중요한 Semantic Segmentation과 같은 태스크에서도 분류기 네트워크의 성능을 효과적으로 향상시키는 데 기여한다.

## 📎 Related Works

본 논문은 Knowledge Distillation (KD) 및 Self-knowledge Distillation (SKD) 분야의 선행 연구들을 검토하고, 제안하는 FRSKD 방법이 이들과 어떻게 차별화되는지 설명한다.

### Knowledge Distillation (KD)

KD의 목표는 사전 학습된 복잡한 Teacher 네트워크의 지식을 간단한 Student 네트워크로 효율적으로 전달하여 학습시키는 것이다.

- **초기 KD**: Hinton et al. [9]은 Teacher 네트워크의 최종 출력 로짓(logit)을 Student 네트워크로 전달하여 지식 증류를 수행하는 방법을 제안하였다.
- **중간 레이어 증류**: 이후 연구들은 Teacher 네트워크의 중간 레이어에서 지식을 활용하는 방법을 소개했다.
  - **Feature-map 레벨 증류**: [26, 37, 14, 34, 16]은 Feature-map 수준에서 로컬리티(locality)를 보존하는 지식 전달 방법을 제안한다. 예를 들어, Teacher 네트워크의 Feature [26], 추상화된 Attention Map [37], 또는 FSP(Factorized Spatial Pooling) 행렬 [34]을 Student가 모방하도록 유도한다.
  - **Penultimate 레이어 증류**: [24, 29, 21, 23, 30]은 마지막에서 두 번째 레이어의 Feature 세트 간 코사인 유사도와 같이 인스턴스 간의 관계를 지식으로 활용한다.
- **기존 KD의 한계**:
    1. 복잡한 Teacher 모델을 사전 학습하는 데 상당한 컴퓨팅 자원이 필요하다.
    2. Teacher 네트워크의 종류에 따라 Student 네트워크의 성능이 달라질 수 있다.

### Self-Knowledge Distillation (SKD)

SKD는 사전 학습된 Teacher 네트워크 없이 Student 네트워크가 스스로의 지식을 활용하여 학습 효과를 높이는 방법이다. SKD는 크게 두 가지 접근 방식으로 나뉜다.

1. **보조 네트워크 기반 접근 방식 (Auxiliary network based approach)**:
    - **BYOT [40]**: 중간 히든 레이어의 Feature를 활용하여 출력을 분류하는 보조 약한 분류기(auxiliary weak classifier) 네트워크 세트를 도입한다. 이 분류기들은 예측 로짓과 실제 Ground-Truth 레이블의 공동 지도(joint supervision)로 훈련된다.
    - **ONE [44]**: 모델 파라미터와 중간 레이어의 추정 Feature를 다양화하기 위해 추가 브랜치를 사용한다. 이 다양성은 앙상블(ensemble) 방식으로 집계되며, 앙상블 출력이 브랜치들이 공유하는 공동 역전파 신호를 생성한다.
    - **한계**: 이러한 보조 네트워크 방식들은 대부분 동일하거나 약한 수준의 분류기 네트워크를 사용하므로, 더 **정제된 지식(refined knowledge)**을 생성하고 전달하는 데 한계가 있다.

2. **데이터 증강 기반 접근 방식 (Data augmentation based approach)**:
    - **DDGSD [32]**: 다르게 변형된(distorted) 인스턴스를 제공하여 분류기 네트워크가 이러한 변형된 데이터에 대해 일관된 예측을 하도록 유도한다.
    - **CS-KD [36]**: 동일 클래스에 속하는 다른 인스턴스의 로짓을 정규화 목적으로 사용하여, 같은 클래스에 대해 유사한 예측을 하도록 한다.
    - **SLA [18]**: 자기 지도 학습(self-supervision) 태스크(예: 입력 회전)와 원래 분류 태스크를 결합하여 레이블 증강을 수행한다. 증강된 인스턴스의 앙상블 출력이 추가적인 지도 신호를 제공한다.
    - **한계**: 데이터 증강은 종종 공간 정보(spatial information)를 보존하지 못할 수 있다 (예: 간단한 이미지 뒤집기는 Feature의 로컬리티를 손상시킬 수 있다). 이로 인해 **Feature-map Distillation을 효과적으로 적용하기 어렵다**.

### 기존 접근 방식과의 차별점

기존 SKD 보조 네트워크는 복잡한 모델에서 정제된 지식을 추출하는 메커니즘을 제공하지 못했으며, 데이터 증강 방식은 데이터의 다양성을 증가시키지만, Feature-map Distillation에 필요한 일관된 로컬리티 모델링을 방해할 수 있다.

본 논문에서 제안하는 **FRSKD**는 이러한 한계점을 극복하기 위해 **보조 self-teacher 네트워크**를 도입한다. 이 self-teacher 네트워크는 단일 인스턴스로부터 **정제된 Feature-map과 soft label을 생성**하며, 이는 기존 SKD 연구들에서 단일 인스턴스로부터 정제된 Feature-map을 생성하는 self-teacher 네트워크를 사용한 첫 시도이다.

### Feature Networks

FRSKD의 self-teacher 네트워크는 객체 탐지 분야의 Feature Network 구조에서 영감을 얻었다.

- **FPN [19]**: Top-down 네트워크를 활용하여 상위 레이어의 추상적 정보와 하위 레이어의 작은 객체 정보를 동시에 활용한다.
- **PANet [20]**: FPN에 추가적인 Bottom-up 네트워크를 도입하여 탐지 레이어와 백본 레이어 간의 짧은 경로 연결을 가능하게 한다.
- **BiFPN [28]**: PANet과 유사하게 Top-down 및 Bottom-up 네트워크를 사용하여 더 효율적인 구조를 제안한다.
- **FRSKD의 차별점**: 본 논문은 BiFPN 구조를 분류 태스크에 적합하도록 변형하여 보조 self-teacher 네트워크로 사용한다. 특히, Feature-map의 깊이에 따라 채널 차원을 조절하여 BiFPN보다 적은 연산으로 Feature-map Distillation을 효율적으로 수행할 수 있도록 설계한다.

## 🛠️ Methodology

본 논문은 Feature Refinement via Self-Knowledge Distillation (FRSKD)라는 새로운 Self-knowledge Distillation 방법을 제안한다. FRSKD는 보조 self-teacher 네트워크를 활용하여 분류기 네트워크에 정제된 Feature-map과 Soft Label을 전달한다.

### 전체 파이프라인 및 시스템 구조

FRSKD의 전체 파이프라인은 그림 2에 나타나 있다. 이 방법은 크게 Classifier Network와 Self-Teacher Network로 구성된다.

- **Classifier Network**: 메인 네트워크로서 실제 분류 또는 세그멘테이션 태스크를 수행한다. 입력 이미지를 받아 Feature Map ($F_1, ..., F_n$)을 생성하고 최종 출력을 예측한다.
- **Self-Teacher Network**: Classifier Network의 Feature Map($F_1, ..., F_n$)을 입력으로 받아 정제된 Feature Map ($T_1, ..., T_n$)과 Soft Label ($\hat{p}_t$)을 생성한다. 이 정제된 Feature Map과 Soft Label은 Classifier Network의 학습을 지도하는 데 사용된다.

두 네트워크는 Ground-Truth Label을 통해 각각 Cross-Entropy Loss를 계산하여 학습하며, Self-Teacher Network가 생성한 지식(Soft Label 및 정제된 Feature Map)은 Classifier Network의 학습에 추가적인 지도(supervision) 신호를 제공한다.

### 주요 구성 요소 및 역할

#### 3.1. Self-Teacher Network

Self-Teacher Network의 주요 목적은 Classifier Network에 정제된 Feature-map과 그에 해당하는 Soft Label을 제공하는 것이다.

- **입력**: Classifier Network의 $n$개 블록에서 추출된 Feature-map $F_1, ..., F_n$.
- **구조**: 객체 탐지에서 사용되는 BiFPN [28]의 구조를 분류 태스크에 맞게 변형하여 사용한다. PANet [20] 및 BiFPN [28]에서 영감을 받은 Top-down 경로와 Bottom-up 경로를 포함한다.

1. **Lateral Convolutional Layers**:
    - Classifier Network의 각 Feature-map $F_i$는 lateral convolutional layer를 통과하여 $L_i$로 변환된다.
    - $$L_i = \text{Conv}(F_i; d_i)$$ (1)
    - 여기서 $\text{Conv}$는 컨볼루션 연산이며, 출력 채널 차원 $d_i$를 가진다.
    - **채널 차원 조절**: 기존 lateral layer와 달리, $d_i$는 해당 Feature-map $F_i$의 입력 채널 차원 $c_i$에 따라 동적으로 설정된다. $d_i = w \times c_i$ (여기서 $w$는 `channel width parameter`).
    - 이 설계는 깊은 레이어일수록 더 많은 채널 차원을 할당하여 Feature-map의 깊이 정보를 포함하게 하고, 동시에 lateral layer의 계산량을 줄이는 효과가 있다.

2. **Top-down Path ($P_i$) 및 Bottom-up Path ($T_i$)**:
    - 이 경로들은 다양한 스케일의 Feature들을 집계한다.
    - **Top-down Path**:
        - $$P_i = \text{Conv}(w_{P_{i,1}} \cdot L_i + w_{P_{i,2}} \cdot \text{Resize}(P_{i+1}); d_i)$$ (2)
        - $P_i$는 Top-down 경로의 $i$-번째 레이어를 나타낸다.
        - $\text{Resize}$는 업샘플링을 위해 bilinear interpolation, 다운샘플링을 위해 max-pooling을 사용한다.
    - **Bottom-up Path**:
        - $$T_i = \text{Conv}(w_{T_{i,1}} \cdot L_i + w_{T_{i,2}} \cdot P_i + w_{T_{i,3}} \cdot \text{Resize}(T_{i-1}); d_i)$$
        - $T_i$는 Bottom-up 경로의 $i$-번째 레이어를 나타내며, Self-Teacher Network가 생성하는 정제된 Feature-map이다.
    - **연결 구조**: BiFPN [28]과 유사하게, 레이어의 깊이에 따라 Forward Pass 연결 구조가 달라진다.
        - 가장 얕은 Bottom-up 레이어 $T_1$과 가장 깊은 Bottom-up 레이어 $T_4$는 효율성을 위해 각각 Lateral Layer $L_1$과 $L_4$를 직접 입력으로 사용한다.
        - 모든 레이어를 연결하기 위해, 마지막 Lateral Layer $L_4$에서 Top-down 경로의 $P_3$로, 그리고 $P_2$에서 Bottom-up 경로의 $T_1$으로 대각선 연결이 추가된다.
    - **Fast Normalized Fusion**: 가중치 $w_P$, $w_T$를 사용하여 Feature 집계 시 [28]의 Fast Normalized Fusion 방식을 적용한다.
    - **효율적인 컨볼루션**: Depth-wise convolution [11]을 사용하여 계산 효율성을 높인다.

- **Self-Teacher Network의 출력**: Bottom-up 경로의 마지막 레이어 $T_n$ 위에 완전 연결(Fully Connected) 레이어를 부착하여 최종 클래스 출력을 예측한다. 이 예측은 Soft Label $\hat{p}_t = \text{softmax}(f_t(x; \theta_t))$ 로 제공된다. 여기서 $f_t$는 Self-Teacher Network를, $\theta_t$는 그 파라미터를 나타낸다.

### 훈련 목표, 손실 함수, 추론 절차

FRSKD는 Self-Teacher Network의 출력인 정제된 Feature-map $T_i$와 Soft Label $\hat{p}_t$를 활용하여 Classifier Network를 학습시킨다.

1. **Feature Distillation Loss ($L_F$)**:
    - Classifier Network가 Self-Teacher Network가 생성한 정제된 Feature-map을 모방하도록 유도한다.
    - 본 논문에서는 Attention Transfer [37] 방식을 채택한다.
    - $$L_F(T, F; \theta_c, \theta_t) = \Sigma_{i=1}^{n} ||\phi(T_i) - \phi(F_i)||_2$$ (3)
    - 여기서 $\phi$는 Feature-map의 공간 정보를 추상화하는 채널-와이즈 풀링 함수와 $L_2$ 정규화를 결합한 함수 [37]이다.
    - $\theta_c$는 Classifier Network의 파라미터이다.
    - $L_F$는 Classifier Network가 Self-Teacher Network의 정제된 Feature-map으로부터 로컬리티(locality)를 학습하게 한다.

2. **Soft Label Distillation Loss ($L_{KD}$)**:
    - Self-Teacher Network가 생성한 Soft Label $\hat{p}_t$를 사용하여 Classifier Network를 증류한다.
    - $$L_{KD}(x; \theta_c, \theta_t, K) = D_{KL}(\text{softmax}(\frac{f_c(x;\theta_c)}{K}) || \text{softmax}(\frac{f_t(x;\theta_t)}{K}))$$ (4)
    - 여기서 $f_c$는 Classifier Network를, $K$는 온도 스케일링 파라미터를 나타낸다.
    - $D_{KL}$은 Kullback-Leibler (KL) 발산이다.

3. **Ground-Truth Cross-Entropy Loss ($L_{CE}$)**:
    - Classifier Network와 Self-Teacher Network는 모두 실제 Ground-Truth Label을 사용하여 학습한다.
    - $L_{CE}(x, y; \theta_c)$는 Classifier Network의 Cross-Entropy Loss이다.
    - $L_{CE}(x, y; \theta_t)$는 Self-Teacher Network의 Cross-Entropy Loss이다.

- **최종 최적화 목표 ($L_{FRSKD}$)**:
  - 위의 세 가지 손실 함수를 통합하여 최종 최적화 목표를 구성한다.
  - $$L_{FRSKD}(x,y;\theta_c,\theta_t,K) = L_{CE}(x,y;\theta_c) + L_{CE}(x,y;\theta_t) + \alpha \cdot L_{KD}(x;\theta_c,\theta_t,K) + \beta \cdot L_F(T,F;\theta_c,\theta_t)$$ (5)
  - 여기서 $\alpha$와 $\beta$는 하이퍼파라미터이다.

- **학습 절차**:
  - Classifier Network와 Self-Teacher Network는 동시에 역전파를 통해 파라미터를 업데이트한다.
  - 모델 붕괴(model collapse) 문제 [22]를 방지하기 위해, 증류 손실인 $L_{KD}$와 $L_F$는 오직 Student Network (Classifier Network)의 파라미터 $\theta_c$ 업데이트에만 적용된다. 즉, Self-Teacher Network의 파라미터 $\theta_t$는 자신의 Cross-Entropy Loss $L_{CE}(x,y;\theta_t)$만을 통해 업데이트된다. Classifier Network의 파라미터 $\theta_c$는 자신의 Cross-Entropy Loss $L_{CE}(x,y;\theta_c)$와 Self-Teacher Network로부터의 증류 손실 $\alpha \cdot L_{KD} + \beta \cdot L_F$를 모두 사용하여 업데이트된다.

## 📊 Results

본 논문은 제안하는 Self-knowledge Distillation 방법인 FRSKD의 효과를 분류(Classification) 및 Semantic Segmentation이라는 두 가지 주요 컴퓨터 비전 태스크에서 다양한 벤치마크 데이터셋을 통해 정량적 및 정성적으로 입증한다.

### 4.1. Classification

- **데이터셋**:
  - **소규모 이미지 분류**: CIFAR-100 [17], TinyImageNet (32x32로 리사이즈).
  - **세밀한 시각 인식 (Fine-Grained Visual Recognition, FGVR)**: CUB200 [31], MIT67 [25], Stanford40 [33], Dogs [13]. 이 데이터셋들은 클래스당 데이터 인스턴스가 적다.
  - **대규모 이미지 분류**: ImageNet [3].
- **백본 네트워크**: CIFAR-100 및 TinyImageNet에는 ResNet18 및 WRN-16-2 [7, 38]를 사용한다. ResNet18은 소형 데이터셋에 맞게 수정되었다. FGVR 태스크에는 표준 ResNet18을 사용하고, ImageNet에는 ResNet18 및 ResNet34를 적용한다.
- **기준선 (Baselines)**: Cross-entropy 손실만을 사용하는 표준 분류기 (Baseline)와 6가지 기존 Self-knowledge Distillation (SKD) 방법들 (ONE [44], DDGSD [32], BYOT [40], SAD [10], CS-KD [36], SLA-SD [18])과 비교한다.
- **측정 지표**: 분류 정확도 (Accuracy), Top-1, Top-5.

- **주요 정량적 결과**:
  - **Table 1 (CIFAR-100 및 TinyImageNet)**: FRSKD는 WRN-16-2 및 ResNet18 백본 모두에서 다른 모든 SKD 방법들보다 일관되게 가장 우수한 성능을 보인다. Feature Distillation을 사용하지 않은 FRSKD\F도 다른 기준선들보다 우수하여, Self-Teacher Network의 Soft Label Distillation 효과를 입증한다. FRSKD는 FRSKD\F보다 더 나은 성능을 보이며 Feature Distillation의 추가적인 이점을 확인시킨다. 또한, 데이터 증강 기반 SKD인 SLA-SD와 FRSKD를 통합한 FRSKD+SLA는 대부분의 실험에서 큰 폭의 성능 향상을 달성하여 FRSKD의 호환성을 보여준다.
  - **Table 2 (FGVR 태스크)**: FRSKD는 모든 FGVR 데이터셋에서 다른 SKD 방법들보다 우수한 분류 정확도를 나타낸다. FRSKD\F 대비 FRSKD의 성능 우위는 이미지 크기가 더 큰 FGVR 데이터셋에서 Feature Distillation의 중요성이 더욱 커짐을 시사한다. FRSKD+SLA는 압도적인 성능 향상을 보이며, FGVR 태스크에서 FRSKD와 데이터 증강 기반 SKD의 결합이 매우 효과적임을 입증한다.
  - **Table 3 (ImageNet)**: 대규모 데이터셋인 ImageNet에서도 FRSKD는 ResNet18과 ResNet34 백본 모두에서 Baseline 대비 Top-1 및 Top-5 정확도를 향상시킨다. 이는 FRSKD가 대규모 실제 환경에서도 효과적임을 보여준다.

### 4.2. Semantic Segmentation

- **데이터셋**: VOC2007 및 VOC2012trainval (학습), VOC2007 test (검증).
- **백본 네트워크**: EfficientDet with stacked BiFPN [28] (Baseline).
- **FRSKD 설정**: Baseline으로 3개의 BiFPN 레이어를 스택하고, Self-Teacher Network로 추가 2개의 BiFPN 레이어를 사용한다.
- **측정 지표**: mIOU (mean Intersection Over Union).

- **주요 정량적 결과**:
  - **Table 4**: FRSKD는 EfficientDet-d0 및 EfficientDet-d1 모델 모두에서 Baseline 대비 mIOU 성능을 크게 향상시킨다 (예: EfficientDet-d0에서 Baseline 79.07% -> FRSKD 80.55%). 이는 FRSKD가 Semantic Segmentation과 같이 로컬 정보 보존이 핵심적인 태스크에서도 효과적인 지식 증류를 통해 성능을 개선할 수 있음을 입증한다.

### 4.3. Further Analyses on FRSKD

- **정성적 Attention Map 비교 (Figure 3, Figure 5)**:
  - Classifier Network와 Self-Teacher Network의 블록별 Attention Map을 비교하여 Self-Teacher가 의미 있는 지식을 전달하는지 확인한다.
  - CUB200 (새 분류), Dogs (개 분류) 데이터셋의 경우, Classifier Network는 주요 객체에 적절히 집중하지 못하는 반면, Self-Teacher Network는 집계된 Feature를 활용하여 주요 객체에 일관되고 명확한 Attention Map을 보인다.
  - MIT67 (실내 장면 인식) 데이터셋에서는 Self-Teacher Network가 장면 클래스(예: 빵집)를 인식하는 데 중요한 단서(예: 빵)에 더 집중하는 모습을 보인다.
  - Figure 5는 학습 진행에 따른 Attention Map의 변화를 보여주는데, 학습 초기에는 두 네트워크 간의 주요 객체 집중도 차이가 크지만, 학습이 진행될수록 두 네트워크 모두 주요 객체에 더 집중하는 경향을 보인다. 이러한 분석은 Self-Teacher Network가 Classifier Network에 효과적으로 정제된 시각적 지도를 제공하고 있음을 시사한다.

- **Feature Distillation 방법 비교 (Table 5)**:
  - FRSKD에 다양한 Feature Distillation 방법 (FitNet [26], Overhaul distillation [8], Attention Transfer [37])을 통합하여 성능 차이를 분석한다.
  - FRSKD의 기본 Feature Distillation 방식인 Attention Transfer가 FitNet 또는 Overhaul distillation 기반의 통합 버전보다 다양한 데이터셋에서 더 높은 정확도를 달성한다.

- **Self-Teacher Network 구조 비교 (Table 6)**:
  - 제안된 Self-Teacher Network의 효율성을 검증하기 위해 다양한 구조를 실험한다.
  - BiFPN은 각 레이어의 채널 차원이 동일하고, BiFPNc는 제안된 방식대로 채널 차원이 레이어 깊이에 따라 다르다.
  - BiFPN (높은 채널 차원 256)이 가장 좋은 성능을 보였으나, 파라미터 수와 FLOPs가 Classifier Network와 유사하거나 더 컸다.
  - 반면, BiFPNc (높은 채널 차원 256)는 BiFPN과 비슷한 성능을 보이면서도 훨씬 적은 계산량(파라미터 및 FLOPs 비율이 더 낮음)을 가졌다. 이는 BiFPNc가 효율적인 Feature-map 정제를 제공함을 의미하며, FRSKD가 Classifier Network를 중복 사용하는 데이터 증강 기반 SKD 방법보다 효율적임을 입증한다.

- **Knowledge Distillation (KD)과의 비교 (Table 7)**:
  - 사전 학습된 Teacher Network를 사용하는 기존 KD 방법들 (FitNet, ATT, Overhaul)과 FRSKD의 성능을 비교한다.
  - Teacher Network로 사전 학습된 ResNet34, Student Network로 ResNet18을 설정하고, 각 KD 방법은 Feature Distillation과 Soft Label Distillation을 모두 활용한다.
  - FRSKD는 대부분의 데이터셋에서 사전 학습된 Teacher Network를 사용하는 기존 KD 방법들보다 우수한 성능을 보인다. 이는 FRSKD가 별도의 대규모 Teacher 모델 없이도 경쟁력 있는 지식 증류 효과를 제공함을 시사한다.

- **데이터 증강과의 훈련 (Table 8)**:
  - FRSKD와 Mixup [39], CutMix [35]와 같은 최신 데이터 증강 방법의 호환성을 검증한다.
  - FRSKD는 이러한 데이터 증강 방법들과 함께 사용될 때 추가적인 큰 성능 향상을 보인다 (예: CutMix 단독 사용 시 CIFAR-100 정확도 79.23% -> FRSKD + CutMix 사용 시 80.49%). 이는 FRSKD가 다른 강력한 학습 기법들과 시너지를 낼 수 있음을 보여준다.

## 🧠 Insights & Discussion

본 논문 FRSKD(Feature Refinement via Self-Knowledge Distillation)는 Self-knowledge Distillation(SKD) 분야의 중요한 발전을 제시하며, 기존 방법론들의 한계를 효과적으로 극복한다.

### 논문에서 뒷받침되는 강점

1. **정제된 Feature-map 활용 능력**: 기존 SKD 방법론들이 보조 네트워크의 한계로 정제된 Feature-map을 생성하거나, 데이터 증강 과정에서 로컬 정보를 손실하는 문제를 안고 있었다. FRSKD는 BiFPN 기반의 보조 self-teacher 네트워크를 도입하여 단일 인스턴스로부터 정제된 Feature-map과 Soft Label을 성공적으로 생성하고 이를 분류기 네트워크에 전달한다. 이는 Semantic Segmentation과 같이 로컬 정보 보존이 중요한 태스크에서 특히 큰 장점으로 작용한다.

2. **효율적인 Self-Teacher Network 설계**: self-teacher 네트워크의 채널 차원을 Feature-map의 깊이에 따라 동적으로 조절($d_i = w \times c_i$)하는 아이디어는 BiFPN과 같은 복잡한 구조의 계산 부담을 크게 줄이면서도 효과적인 Feature 정제 능력을 유지하게 한다. Table 6의 결과는 BiFPNc (채널 차원 조절)가 BiFPN과 유사한 성능을 보이면서도 훨씬 적은 파라미터와 FLOPs를 가짐을 입증한다. 이는 자원 효율적인 SKD 접근 방식을 가능하게 한다.

3. **광범위한 적용 가능성 및 호환성**: FRSKD는 이미지 분류(소규모, 세밀한, 대규모 데이터셋)와 Semantic Segmentation 등 다양한 비전 태스크에서 SOTA 수준의 성능 향상을 보인다. 또한, 기존 데이터 증강 기반 SKD 방법(SLA-SD) 및 Mixup, CutMix와 같은 최신 데이터 증강 기법들과 쉽게 통합되어 추가적인 성능 시너지를 창출한다. 이는 FRSKD가 범용적인 성능 향상 도구로 활용될 수 있음을 시사한다.

4. **효과적인 지식 전달 메커니즘**: Attention Map 정성 분석(Figure 3, 5)을 통해 self-teacher 네트워크가 학습 초기에 분류기 네트워크보다 주요 객체 또는 맥락상 중요한 특징에 더 집중하는 Attention을 형성하고 이를 통해 분류기 네트워크를 지도한다는 점을 보여준다. 이는 self-teacher가 단순히 지식을 복제하는 것을 넘어, "더 나은" 지식을 생성하고 전달하고 있음을 뒷받침한다.

5. **사전 학습 Teacher 불필요**: 기존 Knowledge Distillation의 주요 한계인 대규모 Teacher 모델의 사전 학습 필요성을 없앰으로써, 모델 훈련 프로세스를 간소화하고 컴퓨팅 자원 제약을 완화한다. 동시에 기존 KD 방법들보다 우수한 성능을 달성할 수 있음을 Table 7에서 보여주며 그 효용성을 증명한다.

### 한계, 가정 또는 미해결 질문

1. **하이퍼파라미터 튜닝**: 최종 손실 함수 $L_{FRSKD}$는 $\alpha$와 $\beta$라는 두 개의 가중치 하이퍼파라미터를 포함한다. Sensitivity Analysis(Figure 4)에서 FRSKD가 이 하이퍼파라미터에 대해 강건하다고 언급하지만, 데이터셋마다 최적의 값이 다르다. 실제 적용 시 각 데이터셋에 대한 하이퍼파라미터 튜닝 비용이 발생할 수 있다.

2. **Self-Teacher Network의 복잡도**: 비록 채널 차원 조절을 통해 BiFPN 기반 self-teacher의 효율성을 높였지만, 여전히 보조 네트워크를 추가하는 것은 Baseline 모델에 비해 추가적인 파라미터와 FLOPs를 발생시킨다 (Table 6에서 파라미터/FLOPs 비율이 0.59x/0.68x). 이 추가적인 계산 비용이 자원 제약이 매우 심한 환경(예: 엣지 디바이스)에서 여전히 부담이 될 수 있는지에 대한 심층적인 분석이 필요할 수 있다.

3. **모델 붕괴 방지 메커니즘의 명확성**: "To prevent the model collapse issue [22], FRSKD updates the parameters by the distillation loss, $L_{KD}$ and $L_F$, which is only applied to the student network." 문장은 명시적으로 distillation loss가 teacher network의 파라미터 업데이트에 기여하지 않음을 나타내지만, 왜 이것이 모델 붕괴를 방지하는지에 대한 더 깊은 이론적 또는 실험적 설명은 제한적이다.

4. **Self-Teacher Network의 일반화 능력**: Self-Teacher Network가 단일 인스턴스에서 "정제된" Feature-map을 생성하는 방식에 대한 더 깊은 분석이 있을 수 있다. 어떤 특성을 "정제되었다"고 정의하는지, 그리고 이 정제 과정이 다양한 도메인과 태스크에 걸쳐 어떻게 일반화될 수 있는지에 대한 추가적인 논의가 가능할 것이다.

### 논문에 근거한 간략한 비판적 해석 및 논의사항

FRSKD는 기존 SKD의 핵심적인 약점을 통찰력 있게 분석하고, '정제된 지식'이라는 개념을 도입하여 효과적인 해결책을 제시했다. 특히, BiFPN 구조를 활용하고 채널 차원을 동적으로 조절하는 방식은 계산 효율성과 성능 간의 균형을 잘 맞춘 영리한 설계로 평가할 수 있다. Attention Map 분석은 self-teacher가 단순한 보조 장치를 넘어, 학습 과정에서 중요한 시각적 지도자 역할을 수행함을 정성적으로 잘 보여준다.

하지만, FRSKD가 'Refine Myself by Teaching Myself'라는 제목처럼 스스로를 가르치는 과정에서 발생하는 내재적 한계에 대한 논의는 다소 부족하다. Self-Teacher Network가 생성하는 "정제된 지식"이 결국 Classifier Network의 Feature에 기반한다는 점에서, 근본적으로 Teacher Network의 Feature가 특정 한계를 가질 경우 Self-Teacher 또한 그 한계를 넘어설 수 없을 가능성이 있다. 즉, 네트워크가 처음부터 잘못된 Feature를 학습한다면, Self-Teacher가 이를 "정제"하더라도 그 효과에는 한계가 있을 수 있다. 그럼에도 불구하고, FRSKD는 Self-Knowledge Distillation의 실용성을 크게 높였으며, 향후 'Self-Knowledge Generation' 또는 'Self-Correction'과 같은 방향으로 연구를 확장하는 데 중요한 발판을 마련했다고 볼 수 있다.

## 📌 TL;DR

본 논문은 `Refine Myself by Teaching Myself : Feature Refinement via Self-Knowledge Distillation (FRSKD)`이라는 새로운 Self-knowledge Distillation (SKD) 방법을 제안한다. FRSKD는 기존 SKD 방법론들이 겪는 데이터 증강으로 인한 로컬 정보 손실과 보조 네트워크의 정제된 지식 생성 능력 부족이라는 한계를 극복한다.

**논문의 주요 기여 사항**:

- **보조 Self-Teacher 네트워크 도입**: 분류기 네트워크의 Feature-map을 입력으로 받아 정제된 Feature-map과 Soft Label을 생성하는 보조 self-teacher 네트워크를 활용한다. 이는 단일 인스턴스로부터 정제된 Feature-map을 생성하는 최초의 self-teacher 네트워크 시도이다.
- **효율적인 BiFPN 기반 구조**: Self-teacher 네트워크는 BiFPN 구조를 분류 태스크에 맞게 변형하여, Top-down 및 Bottom-up 경로를 통해 다양한 스케일의 Feature를 효율적으로 집계하고 정제한다. 특히, 채널 차원을 Feature-map 깊이에 따라 동적으로 조절하여 계산 효율성을 극대화한다.
- **Soft Label 및 Feature-map Distillation 통합**: Self-teacher가 생성한 Soft Label에 대한 지식 증류($L_{KD}$)와 정제된 Feature-map에 대한 Feature Distillation($L_F$, Attention Transfer 기반)을 함께 사용하여 분류기 네트워크를 지도한다.
- **광범위한 성능 향상 및 호환성**: FRSKD는 이미지 분류 (CIFAR-100, TinyImageNet, FGVR, ImageNet) 및 Semantic Segmentation (VOC) 등 다양한 비전 태스크에서 Baseline 및 기존 SKD 방법들 대비 우수한 성능을 달성한다. 또한, Mixup, CutMix와 같은 다른 데이터 증강 기법들과도 호환되어 추가적인 성능 시너지를 창출한다.

**이 연구가 실제 적용이나 향후 연구에 중요한 역할을 할 가능성**:
FRSKD는 대규모 사전 학습된 Teacher 모델 없이도 높은 성능을 달성할 수 있어, 제한된 컴퓨팅 자원을 가진 환경(예: 모바일, 엣지 디바이스)에서 딥러닝 모델을 효율적으로 배포하는 데 매우 유용하다. 특히 로컬 정보 보존이 중요한 Semantic Segmentation과 같은 태스크에서도 강점을 보여, 다양한 산업 분야에서의 실제 적용 가능성이 높다. 향후 연구에서는 FRSKD의 self-teacher 메커니즘을 다른 모델 경량화 기법(가지치기, 양자화)과 결합하거나, '정제된 지식'의 정의와 생성 과정을 더욱 일반화하고 최적화하는 방향으로 확장될 수 있을 것이다. 또한, self-teacher가 어떻게 '더 나은' 지식을 생성하는지에 대한 이론적 분석은 Self-Supervised Learning이나 Meta-Learning 분야에도 영감을 줄 수 있다.
