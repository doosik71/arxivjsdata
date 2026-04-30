# Deep Anomaly Detection in Text

Andrei-Marian Manolache (2021)

## 🧩 Problem to Solve

본 논문은 텍스트 데이터 내에서 일반적인 데이터 분포에서 벗어난 데이터 포인트, 즉 이상치(Outlier/Anomaly)를 탐지하는 문제를 다룬다. 텍스트 데이터는 비정형 구조를 가지며 길이가 가변적이기 때문에, 이미지나 정형 데이터에 비해 이상치를 정의하고 탐지하는 것이 까다롭다.

이상치 탐지는 신용카드 부정 결제 탐지, 네트워크 모니터링, 침입 탐지 시스템 등 다양한 보안 및 관리 영역에서 매우 중요한 역할을 한다. 특히, 레이블이 없는 대규모 텍스트 데이터셋에서 효과적으로 이상치를 구분해낼 수 있는 모델의 필요성이 증대되고 있다. 본 연구의 목표는 Transformer 아키텍처와 Self-Supervised Learning(SSL)을 결합하여, 텍스트 데이터에 특화된 end-to-end 딥러닝 기반의 이상치 탐지 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **DATE (Detecting Anomalies in Text using ELECTRA)**라는 새로운 프레임워크를 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

1.  **계층적 Self-Supervision 태스크 도입**: 토큰 수준의 Replaced Token Detection (RTD)과 시퀀스 수준의 Replaced Mask Detection (RMD)이라는 두 가지 pretext task를 동시에 학습시킨다. 이를 통해 모델이 텍스트의 지역적 특성과 전역적 구조를 모두 학습하게 하여 이상치 탐지 능력을 극대화한다.
2.  **효율적인 Pseudo Label (PL) 스코어 설계**: 기존의 $E^3$ Outlier 방식처럼 여러 번의 추론을 거쳐 스코어를 산출하는 대신, 단 한 번의 forward pass를 통해 각 토큰이 '원래의 것(original)'일 확률을 계산하고 이를 평균 내는 효율적인 토큰 수준의 PL 스코어 방식을 제안한다.
3.  **SOTA 성능 달성**: 20Newsgroups 및 AG News 데이터셋에서 기존의 semi-supervised 및 unsupervised 이상치 탐지 모델들보다 우수한 AUROC 성능을 입증하였다.
4.  **실제 적용 가능성 증명**: 텍스트 이상치 탐지 모델을 특정 저자의 텍스트 코퍼스로 학습시킨 후, 다른 저자의 글을 이상치로 처리함으로써 저자 식별(Authorship Detection) 작업에 적용할 수 있음을 보였다.

## 📎 Related Works

논문에서는 크게 고전적 모델과 딥러닝 모델로 나누어 관련 연구를 설명한다.

-   **고전적 모델**: One-Class SVM (OC-SVM)과 Isolation Forest가 대표적이다. OC-SVM은 데이터를 고차원 공간으로 투영하여 원점으로부터 최대 마진을 가지는 초평면을 찾는 방식이며, Isolation Forest는 데이터를 무작위로 분할하여 고립시키는 깊이를 통해 이상치를 판별한다. 이러한 모델들은 단순하지만 딥러닝 기반의 표현 학습 능력이 부족하다는 한계가 있다.
-   **딥러닝 모델**: Autoencoder (AE)나 GAN을 이용해 정상 데이터의 특징을 학습하고 재구성 오차(Reconstruction Error)를 통해 이상치를 찾는 방식이 사용되어 왔다. 
-   **최신 SSL 기반 접근**: 컴퓨터 비전 분야의 $E^3$ Outlier는 이미지에 여러 변환(Rotation, Flip 등)을 가하고 이를 분류하는 pretext task를 통해 이상치를 탐지한다. 텍스트 분야의 CVDD(Context Vector Data Description)는 Self-attention을 이용해 컨텍스트 벡터를 생성하고 코사인 유사도를 측정하는 방식을 사용한다.

DATE는 이러한 기존 방식과 달리, ELECTRA의 판별자(Discriminator) 구조를 활용하여 텍스트의 '부자연스러움'을 직접적으로 측정한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
DATE는 Generator ($G$)와 Discriminator ($D$)로 구성된 구조를 가진다. 학습 단계에서는 두 모델을 함께 사용하지만, 추론(이상치 탐지) 단계에서는 Generator를 버리고 Discriminator만을 사용한다.

### 주요 구성 요소 및 학습 절차
1.  **Generator ($G$)**: Masked Language Model (MLM) 역할을 수행한다. 입력 텍스트의 일부를 마스킹하고, 이를 그럴듯한 대체 토큰으로 채워 넣는다. 본 논문에서는 파라미터화된 Generator보다 무작위로 토큰을 샘플링하는 **Random Generator**가 이상치 탐지 성능을 더 높인다는 점을 발견하였다. 이는 이상치가 정상 데이터와 매우 다르기 때문에, 너무 그럴듯한 텍스트를 생성하면 판별자가 정상 데이터와 이상치를 혼동할 수 있기 때문이다.
2.  **Discriminator ($D$)**: BERT encoder를 기반으로 하며, 두 개의 예측 헤드(Head)를 가진다.
    -   **RTD Head (Token-level)**: 각 토큰이 원래 토큰인지, 아니면 Generator에 의해 교체된 토큰인지 이진 분류한다.
    -   **RMD Head (Sequence-level)**: 입력 텍스트에 적용된 $K$개의 미리 정의된 마스크 패턴 중 어떤 패턴이 사용되었는지 분류한다.

### 손실 함수 (Loss Function)
모델은 다음과 같은 통합 손실 함수를 통해 학습된다.

$$L_{DATE}(\theta_D, \theta_G; x) = \mu L_{RMD}(\theta_D; x) + L_{MLM}(\theta_G; x) + \lambda L_{RTD}(\theta_D; x)$$

여기서 $L_{MLM}$은 Generator의 복원 손실이며, $L_{RTD}$와 $L_{RMD}$는 각각 토큰 수준과 시퀀스 수준의 판별 손실이다. $\mu$와 $\lambda$는 각 손실의 가중치를 조절하는 하이퍼파라미터이다.

### 이상치 탐지 스코어 (Anomaly Score)
추론 시에는 원본 텍스트 $x$를 Discriminator에 입력하고, RTD 헤드에서 출력되는 각 토큰 $i$가 '교체되지 않은 원래의 토큰($m_i=0$)'일 확률을 평균 내어 Pseudo Label (PL) 스코어를 계산한다.

$$PL_{RTD}(x) = \frac{1}{T} \sum_{i=1}^{T} P_D(m_i=0 | x; \theta_D)$$

정상 데이터는 학습 데이터의 분포와 유사하므로 $P_D(m_i=0)$ 값이 높게 나타나며, 이상치는 학습 시 본 적 없는 분포를 가지므로 이 확률값이 낮게 나타난다. 따라서 **낮은 PL 스코어를 가진 샘플을 이상치로 판단**한다.

## 📊 Results

### 실험 설정
-   **데이터셋**: 20Newsgroups (6개 클래스), AG News (4개 클래스).
-   **비교 대상**: Isolation Forest, OC-SVM, CVDD, mSVDD.
-   **지표**: AUROC (Area Under the Receiver Operating Characteristic curve).
-   **설정**: Semi-supervised (한 클래스를 정상으로 간주) 및 Unsupervised (학습 데이터에 일부 이상치 포함) 설정 모두에서 실험을 진행하였다.

### 주요 결과
1.  **정량적 성능**: DATE는 모든 데이터셋 분할에서 경쟁 모델들을 압도하였다. 특히 20Newsgroups에서는 mSVDD 대비 약 4.7% 향상된 성능을 보였으며, AG News에서는 약 6.9% 향상된 성능을 기록하였다.
2.  **Unsupervised 설정**: 학습 데이터에 이상치가 섞여 있는 오염(Contamination) 상황에서도 강건함을 보였다. 데이터 오염도가 10%인 경우에도, 오염도가 0%인 기존 SOTA semi-supervised 모델보다 더 높은 성능을 기록하였다.
3.  **정성적 분석**: 시각화 결과, 특정 도메인(예: 스포츠)에 대해 정치 관련 단어나 기술 관련 단어가 포함된 문장이 낮은 PL 스코어를 기록하며 효과적으로 이상치로 분류됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
-   **상호 보완적 학습**: 토큰 수준(RTD)과 시퀀스 수준(RMD)의 학습을 동시에 진행함으로써, 모델이 언어의 세부적인 문법적 특성과 전체적인 문맥적 구조를 모두 파악하게 되었다. 이는 단일 태스크만 수행했을 때보다 성능이 높고 학습이 안정적이라는 점을 통해 증명되었다.
-   **효율적인 추론**: 기존 SSL 기반 이상치 탐지는 여러 변환된 입력을 반복적으로 넣어 평균을 내야 했으나, DATE는 단 한 번의 forward pass로 토큰 수준의 확신도를 계산함으로써 추론 속도를 획기적으로 개선하였다.

### 한계 및 논의사항
-   **Generator의 선택**: 학습된 Generator보다 Random Generator가 더 좋은 성능을 보였다는 점은 흥미롭다. 이는 이상치 탐지 작업에서 '그럴듯한 가짜'를 만드는 것보다 '명확한 구분선'을 만드는 것이 더 중요하다는 것을 시사한다.
-   **도메인 특이성**: 저자 식별 작업에서는 매우 높은 성능을 보였으나, 원어민/비원어민 구분(ENNTT 데이터셋) 작업에서는 성능이 낮게 나타났다. 이는 문법적 이상치보다 문체적(Stylistic) 이상치를 탐지하는 데 더 최적화되어 있음을 의미한다.

## 📌 TL;DR

본 논문은 ELECTRA 아키텍처를 응용하여 텍스트 내 이상치를 탐지하는 **DATE** 모델을 제안한다. 토큰 수준의 교체 탐지(RTD)와 시퀀스 수준의 마스크 패턴 탐지(RMD)라는 두 가지 Self-supervised 태스크를 동시에 학습하여 텍스트의 정교한 표현을 학습하며, 효율적인 Pseudo Label 스코어링 방식을 통해 추론 속도를 높였다. 실험 결과 20Newsgroups와 AG News 데이터셋에서 SOTA 성능을 달성하였으며, 특히 데이터가 오염된 unsupervised 환경에서도 강건한 성능을 보여 실제 텍스트 이상치 탐지 및 저자 식별 작업에 높은 활용 가능성을 제시하였다.