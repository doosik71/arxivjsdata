# Contrastive Speech Mixup for Low-Resource Keyword Spotting

Dianwen Ng, Ruixi Zhang, Jia Qi Yip, Chong Zhang, Yukun Ma, Trung Hieu Nguyen, Chongjia Ni, Eng Siong Chng, Bin Ma (2023)

## 🧩 Problem to Solve

본 논문은 스마트 기기에서 사용되는 키워드 검출(Keyword Spotting, KWS) 시스템의 저자원(Low-resource) 환경에서의 성능 저하 문제를 해결하고자 한다. 일반적으로 신경망 기반의 KWS 모델이 충분한 오디오 표현(Audio representation)을 학습하기 위해서는 수천 개의 학습 샘플이 필요하지만, 사용자 맞춤형(Personalized) 스마트 기기에 대한 수요가 증가함에 따라 매우 적은 양의 사용자 데이터만으로도 빠르게 적응할 수 있는 모델이 필요해졌다.

기존의 데이터 증강(Data augmentation) 기법들이 일반화 성능을 높이는 데 도움을 주었으나, 음성 전처리에 사용할 수 있는 섭동(Perturbation)의 종류가 제한적이라는 한계가 있다. 특히 Mixup과 같은 기법은 선형 보간을 통해 가상 데이터를 생성하지만, 이 과정에서 공간적으로 모호하거나 부자연스러운 샘플이 생성되어 모델에 혼란을 주고 수렴을 방해할 수 있다는 문제가 존재한다. 따라서 본 연구의 목표는 이러한 Mixup의 부작용을 억제하면서도 저자원 조건에서 모델의 강건성을 높이는 학습 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 기존의 Mixup 증강 기법에 보조적인 대조 손실(Auxiliary contrastive loss)을 결합한 **CosMix** 학습 알고리즘을 제안한 것이다.

CosMix의 중심 아이디어는 원본 샘플(Pre-mixed samples)과 증강된 샘플(Augmented samples) 사이의 상대적 유사도를 최대화하는 제약 조건을 추가하는 것이다. 이를 통해 Mixup으로 인해 발생하는 신호의 모호성을 줄이고, 모델이 더 단순하면서도 풍부한 콘텐츠 기반의 음성 표현을 학습하도록 유도한다. 즉, 노이즈가 섞인 혼합 발화와 깨끗한 원본 발화라는 두 가지 뷰(View)를 통해 모델이 핵심적인 특징을 더 잘 포착하게 만드는 것이 설계의 핵심이다.

## 📎 Related Works

논문에서는 Mixup 알고리즘을 개선하려는 다양한 시도들을 언급한다. 이미지 분류 분야에서는 세그먼트 단위로 교체하는 CutMix나 잠재 공간에서 보간을 수행하는 Manifold Mixup이 제안되었으며, 음성 신호 처리 분야에서는 ASR(Automatic Speech Recognition) 작업을 위해 두 입력 시퀀스를 섞는 MixSpeech나 화자 임베딩 시스템의 안정성을 높이는 L-mix 등이 연구되었다.

기존의 Vanilla Mixup은 단순한 선형 보간을 통해 결정 경계를 부드럽게 만들지만, 앞서 언급한 바와 같이 부자연스러운 인스턴스를 생성하여 모델이 존재하지 않는 공간 정보를 찾으려 할 때 혼란을 야기한다는 한계가 있다. CosMix는 이러한 '부자연스러움'으로 인한 모호성을 해결하기 위해 대조 학습(Contrastive Learning)이라는 명시적인 제약 조건을 도입함으로써 기존 접근 방식과 차별점을 둔다.

## 🛠️ Methodology

### 1. Mixup Augmentation (Baseline)

먼저 기본이 되는 Mixup 과정은 두 개의 무작위 오디오 샘플 $x_i, x_j$와 그에 해당하는 원-핫 레이블 $y_i, y_j$를 선형 보간하여 가상 샘플 $(\tilde{x}, \tilde{y})$를 생성한다.

$$\tilde{x} = \lambda x_i + (1 - \lambda)x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda)y_j$$

여기서 $\lambda$는 $\text{Beta}(\alpha, \alpha)$ 분포에서 추출된 보간 파라미터이다. 이후 STFT를 통해 얻은 특징량 $\tilde{X}$를 모델 $f(\cdot)$에 입력하여 다음과 같은 교차 엔트로피(Cross-Entropy, CE) 손실을 계산한다.

$$L_{mix} = \lambda \cdot CE(f(\tilde{X}), y_i) + (1 - \lambda) \cdot CE(f(\tilde{X}), y_j)$$

### 2. CosMix Learning Algorithm

CosMix는 위 과정에 대조 손실을 추가하여, 혼합된 샘플 $\tilde{X}$와 구성 요소가 된 원본 샘플 $X_r (r \in \{i, j\})$ 사이의 유사도를 높인다.

**시스템 구조 및 절차:**

- **Projector:** 모델의 병목 지점에서 추출된 잠재 벡터 임베딩을 128차원으로 매핑하는 $f_p(\cdot)$ (Linear dense block + ReLU)를 추가한다.
- **Contrastive Loss:** 모든 투영된 임베딩에 $L_2$-norm을 적용한 후, 코사인 유사도를 기반으로 한 평균 제곱 오차(MSE) 형태의 손실을 정의한다.

$$L_{cos}(\tilde{X}, X_r) = -\frac{\langle f_p(\tilde{X}), f_p(X_r) \rangle}{\|f_p(\tilde{X})\|_2 \|f_p(X_r)\|_2}$$

**최종 손실 함수:**
학습 시 Mixup 적용 비율은 50%로 설정하며, $\Lambda_r$이라는 가중치 파라미터를 도입하여 원본 샘플과의 관계를 정의한다.

$$\Lambda_r = \begin{cases} \lambda, & \text{if } r=i \text{ and } r \neq j \\ 1-\lambda, & \text{if } r=j \text{ and } r \neq i \\ 1, & \text{if } r=i=j \end{cases}$$

최종 학습 손실 $L$은 다음과 같이 정의된다.

$$L = L_{mix} + \beta \sum_{r \in \{i,j\}} (\Lambda_r \cdot L_{cos}(\tilde{X}, X_r))$$

여기서 $\beta$는 대조 손실의 기여도를 조절하는 가중치 파라미터이며, 본 논문에서는 $0.5$로 설정되었다.

## 📊 Results

### 실험 설정

- **데이터셋:** Google Speech Command V2 (10개 클래스). 저자원 환경을 시뮬레이션하기 위해 클래스당 학습 데이터를 2.5분(5%), 5분(10%), 10분(20%), 15분(30%), 25분(50%)으로 제한하였다.
- **입력 특징:** 64-dimensional log Mel filterbank (FBank), 해상도 $98 \times 64$.
- **비교 모델:** ResNet18(CNN 기반), KWT-1/3(Transformer 기반), Keyword ConvMixer(CNN 기반).
- **지표:** 테스트 세트에 대한 정확도(Accuracy).

### 정량적 결과

- **전반적 성능 향상:** 모든 모델에서 Baseline 및 Vanilla Mixup 대비 CosMix가 가장 높은 성능을 보였다.
- **저자원 효율성:** 데이터셋 크기가 작을수록 CosMix의 성능 향상 폭이 컸다. 특히 5% 데이터 조건에서 KWT-3 모델은 Baseline 대비 무려 21.7%의 정확도 향상을 보였다.
- **모델별 특성:** Transformer 기반 모델(KWT 시리즈)이 저자원 환경에서 성능 저하가 가장 심했으나, CosMix를 통한 정규화 효과를 가장 크게 얻었다. Convolutional 기반의 Keyword ConvMixer가 가장 높은 절대 성능을 기록했으며, 5% 데이터 조건에서 90%의 정확도를 달성하였다.

### 정성적 결과 및 분석

- **t-SNE 시각화:** KWT-3 모델의 임베딩을 분석한 결과, Baseline은 일부 클래스만 구분했지만 CosMix는 클래스 간의 군집(Cluster)이 가장 명확하게 분리되어 나타났다. 이는 CosMix가 더 정밀하고 콘텐츠가 풍부한 표현을 학습했음을 시사한다.
- **Ablation Study:** $\text{Beta}(10, 10)$ 분포가 $\text{Beta}(0.5, 0.5)$보다 효과적이었으며, 이는 두 오디오가 균등하게 섞인 샘플이 KWS 작업에 더 유리함을 의미한다. 또한 CosMix의 최적 Mixup 비율은 50%로, Vanilla Mixup의 30%보다 높게 나타났다.

## 🧠 Insights & Discussion

본 논문은 Mixup이 제공하는 정규화 효과는 유지하면서, 그로 인해 발생하는 '데이터의 부자연스러움'이라는 부작용을 대조 학습을 통해 효과적으로 제어할 수 있음을 보여주었다. 특히 모델의 복잡도가 높은 Transformer 구조가 적은 데이터 환경에서 취약하다는 점을 확인하였으며, CosMix가 이러한 복잡한 모델의 과적합을 방지하는 강력한 제약 조건으로 작용함을 입증하였다.

다만, 본 연구는 특정 데이터셋(Google Speech Command)과 한정된 모델들을 대상으로 수행되었다. 또한, $\beta$나 $\lambda$의 분포와 같은 하이퍼파라미터 설정에 따라 성능 변동이 발생하는 'Bi-modal distribution' 특성이 관찰되었으므로, 실제 적용 시 정밀한 튜닝이 필요하다는 점이 한계이자 주의사항으로 제시된다. 결과적으로 CosMix는 추가적인 연산 오버헤드가 거의 없으면서도 성능을 높일 수 있어, 실제 스마트 기기의 개인화 KWS 시스템에 적용 가능성이 매우 높다.

## 📌 TL;DR

본 논문은 저자원 환경의 키워드 검출(KWS) 성능을 높이기 위해, 기존 Mixup 기법에 원본-혼합 샘플 간의 유사도를 최대화하는 **대조 손실(Contrastive Loss)**을 결합한 **CosMix** 알고리즘을 제안한다. 실험 결과, 특히 데이터가 매우 적은 조건(클래스당 2.5분)에서 모든 아키텍처의 성능을 일관되게 향상시켰으며, 특히 Transformer 기반 모델의 취약성을 효과적으로 보완하였다. 이 연구는 최소한의 연산 비용으로 저자원 음성 인식 모델의 강건성을 높이는 일반적인 방법론을 제시했다는 점에서 향후 개인화 음성 인터페이스 연구에 중요한 기여를 할 것으로 보인다.
