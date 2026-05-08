# Discriminative Image Generation with Diffusion Models for Zero-Shot Learning

Dingjie Fu, Wenjin Hou, Shiming Chen, Shuhuang Chen, Xinge You, Salman Khan, Fahad Shahbaz Khan (2024)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL) 분야에서 기존의 생성 기반 방식들이 가진 두 가지 핵심적인 문제를 해결하고자 한다. 첫째는 해석 가능성(Interpretability)의 부재이다. 기존의 생성 기반 ZSL 방법론들은 주로 시각적 특징(Visual Feature) 자체를 합성하는 데 집중하였기에, 생성된 데이터가 왜 그러한 형태를 띠는지에 대한 직관적인 설명이나 시각적 분석이 불가능하였다. 둘째는 확장성(Scalability)의 한계이다. 대다수의 기존 방식들이 전문가가 직접 작성한 인간 중심의 시맨틱 프로토타입(Human-annotated semantic prototypes, 예: 속성 리스트)에 의존하고 있어, 일반적인 새로운 장면이나 클래스로 확장할 때 막대한 비용과 시간이 소요되는 문제가 있다.

따라서 본 연구의 목표는 텍스트 프롬프트를 통해 unseen 클래스의 이미지를 직접 생성함으로써 해석 가능성을 확보하고, 인간의 주석 없이도 클래스 이름으로부터 유도된 시맨틱 정보를 활용하여 일반화 성능을 높이는 Discriminative Image Generation (DIG-ZSL) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사전 학습된 Diffusion Model을 활용하여 ZSL을 위한 '판별 가능한 이미지'를 생성하는 것이다. 단순히 이미지를 생성하는 것에 그치지 않고, 다음과 같은 설계 전략을 통해 ZSL 분류기에 최적화된 데이터를 생성한다.

1. **Category Discrimination Model (CDM) 도입**: seen 클래스의 데이터와 클래스 이름 기반의 텍스트 임베딩만을 사용하여 카테고리 판별 모델을 먼저 학습시킨다. 이 모델은 이후 생성될 이미지가 해당 클래스의 특징을 잘 반영하고 있는지 판단하는 가이드 역할을 한다.
2. **Discriminative Class Token (DCT) 학습**: 각 unseen 클래스에 대해, CDM의 가이드를 받아 최적화된 특수 토큰인 DCT를 학습한다. 이 토큰은 텍스트-이미지 확산 모델의 프롬프트에 삽입되어, 생성된 이미지가 실제 이미지 분포와 가깝고 클래스 간 판별력이 높도록 유도한다.
3. **이미지 기반 ZSL 파이프라인**: 학습된 DCT를 통해 생성된 고품질의 unseen 클래스 이미지들을 실제 seen 클래스 이미지와 함께 사용하여 최종적인 ZSL 분류기를 학습시킨다.

## 📎 Related Works

ZSL 연구는 크게 임베딩 기반 방식과 생성 기반 방식으로 나뉜다. 임베딩 기반 방식은 시각적 특징을 시맨틱 공간으로 투영하여 정렬하지만, seen 클래스에 과적합(Overfitting)되는 경향이 있다. 이를 해결하기 위해 등장한 생성 기반 방식은 GAN이나 VAE를 이용해 unseen 클래스의 특징을 합성하여 데이터 증강을 수행한다. 그러나 앞서 언급했듯이 특징 생성 방식은 해석력이 떨어지고 인간의 주석에 의존하는 한계가 있다.

최근의 Text-to-Image Diffusion Model(예: Stable Diffusion)은 매우 사실적인 이미지를 생성할 수 있게 되었으나, 이는 주로 이미지의 품질이나 텍스트 정렬도에 치중되어 있어 분류 작업과 같은 다운스트림 태스크에서의 유효성은 충분히 검증되지 않았다. 본 연구는 Textual Inversion이나 DREAM-OOD와 같이 임베딩 공간에서 새로운 개념을 학습하는 접근법과 유사하지만, 모든 클래스의 데이터가 아닌 오직 seen 클래스의 데이터만을 활용하여 CDM을 구축하고 이를 통해 DCT를 최적화한다는 점에서 ZSL 설정에 충실한 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

DIG-ZSL은 크게 세 단계의 파이프라인으로 구성된다: $\text{CDM 학습} \rightarrow \text{DCT 최적화} \rightarrow \text{ZSL 분류기 학습}$.

### 1. Category Discrimination Model (CDM)

CDM은 frozen된 ViT 백본을 통해 이미지 $x_i$에서 시각적 특징 $f_i \in \mathbb{R}^{d_v \times 1}$를 추출하고, 학습 가능한 MLP를 통해 이를 시맨틱 공간의 투영 특징 $a_i \in \mathbb{R}^{d_t \times 1}$로 변환한다. 이때 CLIP의 텍스트 인코더에서 추출한 클래스 시맨틱 프로토타입 $v_c$를 사용하여 시각-시맨틱 정렬을 수행한다. CDM은 오직 seen 클래스의 데이터로만 학습되어, 이후 DCT 학습 단계에서 unseen 클래스 이미지의 판별력을 측정하는 기준으로 사용된다.

### 2. Discriminative Class Token (DCT)

각 unseen 클래스에 대해, 텍스트 인코더의 어휘집에 없는 새로운 토큰 $S^*$를 도입하고 그 임베딩 $e^*$를 최적화한다. 텍스트 프롬프트는 $p = \text{"A photo of } S^* \text{ [name]"}$ 형태로 구성된다.

생성기 $G$로부터 생성된 이미지 $ex_i$의 시맨틱 특징 $v^{gen}$을 CDM으로 추출한 후, CLIP 프로토타입 $v_c$와의 코사인 유사도를 통해 클래스 점수 $s^c_i$를 계산한다:
$$s^c_i = \cos(v^{gen}, v_c)$$

이 점수를 바탕으로 다음과 같은 Cross-Entropy 손실 함수를 사용하여 $e^*$를 최적화한다:
$$L_{ce} = -\mathbb{E} \left[ \log \frac{\exp(s^c_i)}{\sum_{c' \in C_u} \exp(s^{c'}_i)} \right]$$
이 과정은 오직 unseen 클래스들 사이의 판별력을 높이는 방향으로 진행된다.

### 3. ZSL Classifier 및 추론

최적화된 DCT를 사용하여 각 unseen 클래스당 일정 수(예: 100장)의 이미지를 생성한다. 이 생성된 데이터와 실제 seen 데이터를 합쳐 최종 분류기 $f_{zsl}$을 학습시킨다. 특히 Generalized ZSL (GZSL) 설정에서는 seen 클래스로의 편향(Bias)을 줄이기 위해 다음과 같은 보정 계수 $\lambda$를 도입하여 최종 클래스 $c^*$를 예측한다:
$$c^* = \text{argmax}_{c \in C} (o - \lambda \mathbb{I}[c \in C_s])$$
여기서 $o$는 softmax 이후의 로짓 값이며, $\mathbb{I}[c \in C_s]$는 클래스 $c$가 seen 클래스일 때 1인 지시 함수이다.

## 📊 Results

### 실험 설정

- **데이터셋**: AWA2, CUB, FLO, SUN (4종)
- **평가 지표**: CZSL에서는 Top-1 Accuracy ($\text{acc}$), GZSL에서는 seen/unseen 성능의 조화 평균인 Harmonic Mean ($H$)을 사용한다.
- **기준선**: 비인간 주석 기반(Nonhuman-annotated) 방식(Glove, MPNet, I2DFormer+ 등) 및 인간 주석 기반 방식과 비교한다.

### 주요 결과

1. **비인간 주석 기반 방식 대비 압도적 성능**: CZSL 설정에서 DIG-ZSL은 AWA2(90.1%), CUB(69.7%), FLO(70.9%), SUN(68.8%)의 정확도를 기록하며, 기존 SOTA인 I2DFormer+보다 최소 12.8%에서 최대 30.7%까지 성능을 향상시켰다.
2. **GZSL 성능**: GZSL에서도 강력한 성능을 보였으며, 특히 FLO($H=81.7\%$)와 SUN($H=48.5\%$)에서 비약적인 상승을 보였다. 또한 CLIP이나 CoOp과 같은 대규모 시각-언어 모델 기반 ZSL보다 높은 성능을 기록하였다.
3. **인간 주석 기반 방식과의 비교**: AWA2와 SUN에서는 인간의 세밀한 속성 주석을 사용한 기존 generative 방법들보다 더 높은 성능을 보였으나, 매우 세밀한 분류가 필요한 CUB 데이터셋에서는 다소 밀리는 모습을 보였다.
4. **이미지 품질**: FID 스코어 측정 결과, vanilla Stable Diffusion보다 DIG-ZSL이 생성한 이미지가 실제 데이터 분포에 더 가깝고 품질이 높음을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Diffusion Model을 ZSL에 접목하여 '특징 생성'에서 '이미지 생성'으로 패러다임을 전환함으로써 해석 가능성과 성능을 동시에 잡았다. 특히 CDM과 DCT라는 구조를 통해, 단순히 그럴듯한 이미지를 만드는 것이 아니라 '분류기에 유용한' 판별적 특징을 가진 이미지를 생성해냈다는 점이 고무적이다.

**강점 및 한계**:

- **강점**: 인간의 수동 주석 없이 클래스 이름만으로 높은 성능을 낼 수 있어 확장성이 매우 뛰어나며, 생성된 이미지를 통해 모델이 무엇을 학습하는지 시각적으로 확인할 수 있다.
- **한계**: CUB와 같은 초미세 분류(Fine-grained) 데이터셋에서 성능이 상대적으로 낮은데, 이는 Diffusion Model이 매우 복잡하고 미세한 세부 특징(예: 새의 부리 모양 등)을 정확하게 생성하는 데 여전히 어려움이 있음을 시사한다.
- **비판적 해석**: 성능 향상에 있어 $\lambda$(보정 계수)와 $\gamma$(DCT 학습 조기 종료 임계값)와 같은 하이퍼파라미터에 대한 의존도가 높다. 데이터셋마다 이 값들을 수동으로 설정해야 한다는 점은 실용적 관점에서 개선이 필요한 부분이다.

## 📌 TL;DR

본 연구는 Diffusion Model을 이용하여 unseen 클래스의 판별 가능한 이미지를 생성하고 이를 통해 ZSL을 수행하는 **DIG-ZSL** 프레임워크를 제안한다. CDM(카테고리 판별 모델)의 가이드를 통해 최적화된 DCT(판별 클래스 토큰)를 학습함으로써, 인간의 수동 주석 없이도 기존의 특징 생성 기반 ZSL 모델들을 크게 상회하는 성능을 달성하였다. 이 연구는 ZSL의 데이터 증강 방식을 시각적 특징 레벨에서 이미지 레벨로 확장하여 해석 가능성을 높였으며, 향후 생성 AI를 활용한 제로샷 학습 연구에 중요한 방향성을 제시한다.
