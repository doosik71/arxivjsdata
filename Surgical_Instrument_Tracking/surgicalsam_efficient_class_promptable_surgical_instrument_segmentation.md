# SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation

Wenxi Yue, Jing Zhang, Kun Hu, Yong Xia, Jiebo Luo, Zhiyong Wang (2024)

## 🧩 Problem to Solve

본 논문은 수술 도구 분할(Surgical Instrument Segmentation, SIS) 작업에 Segment Anything Model (SAM)을 적용할 때 발생하는 두 가지 핵심 문제를 해결하고자 한다.

첫째, **자연물과 수술 도구 사이의 도메인 간극(Domain Gap)** 문제이다. SAM은 방대한 자연 이미지 데이터로 사전 학습되었으나, 수술 도구는 특수한 외형, 복잡한 해부학적 배경, 그리고 클래스 간의 매우 유사한 특징을 가지고 있다. 이로 인해 SAM을 제로샷(Zero-shot) 방식으로 그대로 적용할 경우 일반화 성능이 현저히 떨어진다.

둘째, **명시적 프롬프트(Explicit Prompt)에 대한 높은 의존성** 문제이다. SAM이 정확한 분할을 수행하기 위해서는 정밀한 포인트(Point)나 바운딩 박스(Bounding Box) 위치 정보가 필수적이다. 이를 위해 수작업으로 가이드를 제공하거나 매우 정교한 전용 검출기(Specialist Detector)를 앞단에 배치해야 하며, 이는 전체 파이프라인을 복잡한 다단계(Multi-stage) 구조로 만든다. 또한, 작은 프롬프트 노이즈(Jitter)만으로도 성능이 급격히 저하되는 취약성이 존재한다.

따라서 본 논문의 목표는 명시적인 좌표 정보 없이 클래스 정보만으로 수술 도구를 분할할 수 있는 효율적인 튜닝 방법론인 **SurgicalSAM**을 제안하는 것이다.

## ✨ Key Contributions

SurgicalSAM의 핵심 아이디어는 정밀한 좌표 기반 프롬프트 대신, 학습 가능한 **클래스 프로토타입(Class Prototype)**을 사용하여 SAM을 효율적으로 튜닝하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Prototype-based Class Prompt Encoder 제안**: 클래스 프로토타입을 통해 이미지 임베딩을 활성화하고, 이를 통해 SAM의 Mask Decoder가 이해할 수 있는 Dense 및 Sparse 프롬프트 임베딩을 직접 생성한다. 이를 통해 명시적인 포인트나 박스 프롬프트 없이 클래스 ID만으로 분할이 가능한 end-to-end 파이프라인을 구축하였다.
2. **Contrastive Prototype Learning 도입**: 수술 도구들 사이의 낮은 클래스 간 분산(Inter-class variance) 문제를 해결하기 위해 대조 학습(Contrastive Learning)을 적용하였다. 이를 통해 각 도구 클래스를 대표하는 프로토타입의 변별력을 높여 세밀한 클래스 구분이 가능하게 하였다.
3. **효율적인 파라미터 튜닝(Efficient Tuning)**: 거대한 Image Encoder는 동결(Frozen)시킨 채, 경량화된 Prompt Encoder와 Mask Decoder만을 튜닝함으로써 학습 효율성을 극대화하고 적은 데이터셋에서도 높은 일반화 성능을 확보하였다.

## 📎 Related Works

### 수술 도구 분할 (SIS) 연구

기존의 SIS 연구는 크게 두 가지 패러다임으로 나뉜다.

- **픽셀 분류(Pixel Classification)**: U-Net 기반의 TernausNet 등이 대표적이며, 픽셀 단위로 클래스 확률을 예측한다. 하지만 하나의 도구가 여러 클래스로 지정되는 공간적 클래스 불일치 문제가 발생한다.
- **마스크 분류(Mask Classification)**: ISINet, TraSeTR, MATIS 등은 마스크를 먼저 예측하고 이에 클래스 라벨을 부여하는 방식을 취하며, 공간적 불일치 문제를 완화하였다.
- **한계**: 이러한 전용 모델들은 전체 파라미터를 학습시켜야 하므로 비효율적이며, 수술 데이터셋의 규모가 작아 일반화 성능이 낮다는 단점이 있다.

### Segment Anything Model (SAM) 및 의료 영상 적용

SAM은 강력한 제로샷 능력을 갖추고 있으나, 의료 영상 분야에서는 도메인 간극으로 인해 성능이 저하된다. 이를 해결하기 위해 도메인 특화 파인튜닝이나 어댑터(Adapter)를 사용하는 연구들이 진행되었으나, 여전히 정밀한 포인트/박스 프롬프트가 필요하거나 클래스 간 변별력이 부족한 범용 프롬프트 임베딩을 사용하는 한계가 있었다.

## 🛠️ Methodology

### 전체 시스템 구조

SurgicalSAM은 입력 이미지 $I$와 타겟 클래스 $c$를 받아 해당 클래스의 마스크 $M^{(c)}$를 예측한다. 전체 프로세스는 다음과 같은 수식으로 정의된다.

$$F^I = E^I(I)$$
$$T^{(c)}_D, T^{(c)}_S = E^{CP}(F^I, B, c)$$
$$M^{(c)} = D^M(F^I, [T^{(c)}_D, T^{(c)}_S, T^O])$$

여기서 $E^I$는 동결된 Image Encoder, $E^{CP}$는 제안된 Prototype-based Class Prompt Encoder, $D^M$은 튜닝 가능한 Mask Decoder, $B$는 클래스 프로토타입 뱅크, $T^O$는 SAM의 학습 가능한 출력 토큰이다.

### Prototype-based Class Prompt Encoder

이 모듈은 이미지 임베딩 $F^I \in \mathbb{R}^{h \times w \times d}$와 클래스 프로토타입 $B \in \mathbb{R}^{C \times d}$ 사이의 유사도를 이용해 프롬프트를 생성한다.

1. **클래스 활성화 특징(Class-activated Feature) 생성**:
   모든 클래스 $k$에 대해 이미지 임베딩과 프로토타입의 내적(Dot product)을 통해 유사도 행렬 $S$를 계산한다.
   $$S^{(k)} = F^I \times B^{(k)}, \quad \text{for } k \in \{1, \dots, C\}$$
   이후 이를 공간적 어텐션으로 사용하여 클래스별 활성화 특징 $F^{(k)}_I$를 생성한다.
   $$F^{(k)}_I = F^I \circ S^{(k)} + F^I$$

2. **Dense Prompt Embedding ($T^{(c)}_D$) 생성**:
   타겟 클래스 $c$의 활성화 특징 $F^{(c)}_I$를 2층 MLP(Multilayer Perceptron)에 통과시켜 생성한다.
   $$T^{(c)}_D = g^D(\text{ReLU}(f^D(F^{(c)}_I)))$$

3. **Sparse Prompt Embedding ($T^{(c)}_S$) 생성**:
   모든 클래스의 활성화 특징 $F^C_I$를 MLP에 통과시켜 클래스 무관(Positivity-agnostic) 임베딩 $\hat{T}^{(k)}_S$를 먼저 생성한다. 이후 타겟 클래스 $c$에는 양성 임베딩 $\lambda^+$를, 나머지 클래스에는 음성 임베딩 $\lambda^-$를 더해 클래스 구분 가능(Positivity-aware) 임베딩을 완성한다.
   $$T^{(c)}_S = \text{concat}(\{\hat{T}^{(k)}_S + \mathbb{1}(k=c)\lambda^+ + (1-\mathbb{1}(k=c))\lambda^-\})$$

### Contrastive Prototype Learning

유사한 외형의 도구들을 구분하기 위해 InfoNCE 손실 함수에서 영감을 받은 프로토타입 대조 손실($L^{PCL}$)을 도입하였다.
먼저, 정답 마스크 $G^{(c)}$를 이용하여 이미지 임베딩에서 전경(Foreground) 특징의 평균을 내어 SAM 기반 클래스 임베딩 $v^{(c)}$를 추출한다.

$$v^{(c)} = \frac{\sum_{i} (F^I \circ G^{(c)})}{\sum_{i} G^{(c)}}$$

이후 다음과 같은 손실 함수를 통해 프로토타입 $B^{(k)}$가 동일 클래스의 $v^{(k)}$와는 가까워지고, 타 클래스의 $v^{(q)}$와는 멀어지도록 학습한다.

$$L^{PCL} = -\frac{1}{C} \sum_{k=1}^{C} \log \frac{\exp(B^{(k)} \cdot v^{(k)} / \tau)}{\sum_{q=1}^{C} \exp(B^{(k)} \cdot v^{(q)} / \tau)}$$

### 학습 절차 및 손실 함수

Image Encoder는 동결하고, Prompt Encoder와 Mask Decoder만 업데이트한다. 최종 손실 함수는 분할을 위한 Dice Loss와 프로토타입 학습을 위한 Contrastive Loss의 합으로 구성된다.

$$L = L_{DICE} + L^{PCL}$$

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis2017, EndoVis2018
- **평가 지표**: Challenge IoU, IoU, mean class IoU (mc IoU)
- **비교 대상**: 전용 모델(TernausNet, MATIS Frame 등), SAM 기반 제로샷 모델(Mask2Former+SAM, Track Anything, PerSAM 등)

### 주요 결과

- **정량적 성과**: SurgicalSAM은 제로샷 SAM 기반 모델들을 압도하였으며, SOTA 전용 모델들과 대등하거나 더 높은 성능을 보였다. 특히 EndoVis2018에서 mc IoU 기준 높은 성능 향상을 보였는데, 이는 파운데이션 모델의 일반 지식이 소규모 데이터셋의 클래스 불균형 문제를 완화하는 사전 지식으로 작용했음을 시사한다.
- **효율성**: 튜닝 가능한 파라미터 수가 단 4.65M개로, 전용 모델(MATIS Frame: 68.72M)이나 타 SAM 기반 모델(MaskTrack-RCNN+SAM: 57.67M)보다 훨씬 적다.
- **일반화 능력**: Cross-dataset 실험(한 데이터셋에서 학습 후 다른 데이터셋에서 평가)에서 SOTA 전용 모델인 MATIS Frame보다 일관되게 높은 성능을 보여, 새로운 데이터 분포에 대한 적응력이 뛰어남을 입증하였다.
- **복잡도 분석**: 학습 속도가 MATIS Frame 대비 10배 이상 빠르며, GPU 메모리 사용량은 1/6 수준으로 매우 효율적이다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 SAM의 강력한 마스크 생성 능력을 유지하면서도, 수술 도구라는 특수 도메인 지식을 **클래스 프로토타입**이라는 형태로 효율적으로 주입하였다는 점이 매우 뛰어나다. 특히 좌표 기반의 명시적 프롬프트를 제거함으로써 프롬프트 노이즈에 대한 강건성(Robustness)을 확보하고, 사용자(외과의)가 단순히 클래스 ID만으로 조작할 수 있는 인터랙티브한 가능성을 열었다.

### 한계 및 논의사항

- **데이터셋 의존성**: 실험이 EndoVis 데이터셋에 한정되어 있어, 더 다양한 수술 환경이나 다른 장비가 사용되는 환경에서의 범용성은 추가 검증이 필요하다.
- **프로토타입 초기화**: 프로토타입을 표준 정규 분포로 초기화하여 학습하였으나, CLIP과 같은 사전 학습된 텍스트-이미지 임베딩을 초기값으로 사용했을 때의 성능 향상 가능성이 존재한다. (실제로 텍스트 프롬프트 기반 베이스라인보다 대조 학습된 프로토타입의 성능이 훨씬 높았음이 확인되었다.)

## 📌 TL;DR

SurgicalSAM은 SAM을 수술 도구 분할에 최적화하기 위해 **명시적 좌표 프롬프트 없이 클래스 프로토타입만으로 작동하는 효율적인 튜닝 프레임워크**이다. Contrastive Prototype Learning을 통해 도구 간의 유사성 문제를 해결하였으며, 매우 적은 수의 파라미터(4.65M)만 튜닝하고도 SOTA 수준의 성능과 뛰어난 일반화 능력을 달성하였다. 이 연구는 파운데이션 모델을 의료 특수 도메인에 맞게 경량화하여 적응시키는 효과적인 방법론을 제시하였다.
