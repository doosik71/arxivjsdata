# A Survey of Anomaly Detection in In-Vehicle Networks

Ozdemir et al. (2024)

## 🧩 Problem to Solve

현대 자동차는 안전 필수 작업을 제어하는 다수의 전자 제어 장치(Electronic Control Units, ECU)로 구성되어 있으며, 이들은 차량 내부 네트워크를 통해 정보를 교환한다. 특히 Controller Area Network (CAN bus)는 가장 널리 사용되는 통신 프로토콜이다. 하지만 CAN 버스는 메시지를 전체 네트워크에 브로드캐스트하는 특성과 데이터 필드의 길이가 제한적이라는 점 때문에 암호화 및 인증 구현이 어렵다.

이러한 구조적 취약성으로 인해 차량의 물리적 부품 결함이나 외부의 악의적인 공격(Cyber-attacks)이 발생할 경우, CAN 트래픽에 이상 현상(Anomaly)이 나타나며 이는 차량의 정상적인 작동을 저해하고 운전자의 안전을 심각하게 위협할 수 있다. 따라서 차량 네트워크 내의 이상 현상을 조기에 탐지하는 것은 차량 안전 보장을 위해 매우 중요하다. 본 논문의 목표는 CAN 버스를 중심으로 한 차량 내 네트워크 이상 탐지 기법과 사용된 데이터셋에 대해 학술적 수준의 포괄적인 리뷰를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 CAN 버스 이상 탐지 분야의 연구 동향을 체계적으로 정리하여 연구자들에게 가이드를 제공하는 것이다. 구체적인 기여 사항은 다음과 같다.

- **포괄적인 방법론 및 데이터셋 리뷰**: 전통적인 통계적 방법부터 최신 딥러닝 기법에 이르기까지 CAN 버스 이상 탐지에 사용된 다양한 방법론과 공개 데이터셋을 광범위하게 분석하였다.
- **학습 패러다임별 분류**: 이상 탐지 기법을 지도 학습(Supervised), 준지도 학습(Semi-supervised), 비지도/자기지도 학습(Unsupervised/Self-supervised)의 네 가지 패러다임으로 분류하여 각 방법론의 강점과 약점을 대조 분석하였다.
- **시계열 분석 관점의 확장**: CAN 버스 데이터가 본질적으로 시계열 데이터라는 점에 착안하여, 일반적인 시계열 이상 탐지 기법과 데이터셋에 대한 요약을 제공함으로써 CAN 버스 특화 접근 방식과의 공통점 및 차이점을 이해하도록 돕는다.
- **최신 트렌드 분석**: 단순 탐지를 넘어 데이터 프라이버시 보호를 위한 Federated Learning과 AI 의사결정의 투명성을 위한 Explainability(설명 가능성)와 같은 최신 연구 방향을 제시한다.

## 📎 Related Works

기존에도 차량 내 이상 탐지 및 침입 탐지 시스템(IDS)에 관한 여러 서베이 연구들이 존재하였다. 일부 연구들은 사이버 보안 위협의 분류 체계(Taxonomy)를 제시하거나, 특정 공격 전략 및 탐지 알고리즘에 집중하였다. 또한, ECU 하드웨어 특성 기반 탐지나 CAN 패킷 기반 탐지로 나누어 분석한 사례도 있다.

그러나 본 논문은 기존 연구들이 다음과 같은 한계를 가지고 있다고 지적한다. 첫째, 최신 딥러닝 기법에 대한 심층적인 리뷰가 부족하며, 둘째, 이상 탐지 성능 평가의 핵심인 데이터셋에 대한 포괄적인 분석이 결여되어 있다. 본 논문은 2015년부터 2023년까지 Scopus 데이터베이스에서 선정된 85편의 논문을 분석함으로써, 최신 딥러닝 기술과 데이터셋의 효용성을 함께 다루어 기존 서베이 논문들과 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. CAN 버스 데이터 준비 (Data Preparation)

CAN 프레임은 중재 필드(Arbitration field)의 ID와 실제 데이터가 담긴 데이터 필드(Data field)로 구성된다. 연구자들은 이 데이터를 다음과 같은 방식으로 전처리하여 모델의 입력값으로 사용한다.

- **ID 및 페이로드 기반 특징 추출**: 슬라이딩 윈도우를 적용하여 패킷 수, 통계적 모멘트, ID 분포 기반의 엔트로피 등을 계산한다.
- **저수준 인코딩**: ID와 데이터를 비트(Bit) 수준이나 바이트(Byte) 수준으로 처리하거나, 16진수 값을 10진수로 변환하여 사용한다.
- **이미지 변환**: ID 시퀀스를 2D 이미지 형태로 변환하여 CNN의 입력으로 사용한다.
- **물리 계층 신호 분석**: 오실로스코프 등을 통해 전압 차이(Voltage difference)와 같은 물리적 신호를 추출하여 시계열 데이터로 활용한다.

### 2. 이상 탐지 방법론 분류

논문은 이상 탐지 기법을 네 가지 학습 패러다임으로 구분하여 설명한다.

- **통계적 방법 (Statistical Methods)**: 메시지 전송 주기(Time intervals), 주파수 분석(Wavelet transformation), ECU 고유의 핑거프린트(Clock skew, Drift) 등을 이용하여 정상 범위에서 벗어난 데이터를 탐지한다.
- **지도 학습 (Supervised Learning)**: 레이블이 지정된 데이터를 사용하여 분류 문제로 접근한다. Random Forest, SVM과 같은 전통적 ML부터 MLP, CNN, LSTM 및 이들의 하이브리드 모델(예: LSTM-ResNet)이 사용된다.
- **준지도 학습 (Semi-supervised Learning)**: 주로 정상 데이터만을 학습하여 정상 패턴을 정의하고, 여기서 벗어나는 데이터를 이상치로 간주한다. GAN의 생성자를 통해 정상 데이터를 모사하거나, Autoencoder의 재구성 오차(Reconstruction error)를 이상치 점수로 활용하는 방식이 대표적이다.
- **비지도 및 자기지도 학습 (Unsupervised & Self-supervised Learning)**: 레이블 없이 데이터 자체의 구조를 학습한다. LSTM을 이용해 다음 시퀀스를 예측하고 예측 오차를 측정하거나, Triplet Loss를 통해 정상-이상 샘플 간의 거리를 최대화하는 임베딩 공간을 학습한다. 또한 K-means clustering이나 Isolation Forest 등이 활용된다.

### 3. 평가 지표 (Evaluation Metrics)

이상 탐지는 클래스 불균형(Imbalance)이 심하므로 단순 정확도보다는 다음과 같은 지표들을 종합적으로 평가한다. 여기서 $TP, FP, TN, FN$은 각각 True Positive, False Positive, True Negative, False Negative를 의미한다.

- **Accuracy**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall (Sensitivity)**: $\text{Recall} = \frac{TP}{TP + FN}$
- **F-score**: $\text{F}_\beta = (1 + \beta^2) \times \frac{\text{precision} \times \text{recall}}{(\beta^2 \times \text{precision}) + \text{recall}}$
- **False Negative Rate (FNR)**: $\text{FNR} = \frac{FN}{TP + FN}$
- **False Positive Rate (FPR)**: $\text{FPR} = \frac{FP}{TN + FP}$
- **True Negative Rate (TNR)**: $\text{TNR} = \frac{TN}{TN + FP}$
- **AUC (Area Under the ROC Curve)**: 모델의 클래스 구분 능력을 0에서 1 사이의 값으로 나타낸다.

## 📊 Results

본 논문은 다양한 공개 데이터셋(OTIDS, Car Hacking Dataset, SynCAN, ROAD 등)에서 평가된 고성능 모델들을 정리하여 제시한다.

- **정량적 결과**: 지도 학습 기반 모델들은 Car Hacking 및 OTIDS 데이터셋에서 매우 높은 정확도와 TPR을 보였다. 특히 LSTM-ResNet, ANN-based AD 등의 모델이 높은 성능을 기록하였다.
- **준지도/비지도 학습의 성과**: GAN-based GIDS나 LSTM-AE 같은 모델들은 레이블이 부족한 상황에서도 높은 AUC(0.97~0.99)를 기록하며 효과적인 탐지 능력을 입증하였다.
- **데이터셋 특성**: 분석 결과, 대부분의 공개 데이터셋이 외부에서 유도된 '침입(Intrusion)' 시나리오에 집중되어 있으며, 차량 내부의 자연스러운 '결함(Fault)'이나 '고장' 상황을 반영한 데이터셋은 매우 부족함을 확인하였다.

## 🧠 Insights & Discussion

### 1. 학습 패러다임의 트레이드-오프

지도 학습은 높은 정확도를 보이지만, 학습 데이터에 포함되지 않은 **미지의 공격(Unseen anomalies)**에 취약하며 과적합(Overfitting) 위험이 크다. 반면, 준지도 및 자기지도 학습은 정상 데이터의 일반적인 표현(Generic representation)을 학습하므로 미지의 이상 현상을 탐지하는 데 더 유리하다.

### 2. 자원 제약 및 계산 효율성

차량 내 ECU는 컴퓨팅 자원이 제한적이므로, 무거운 모델보다는 경량화된 ML 모델, 효율적인 특징 공학(Feature engineering), 모델 압축 및 전이 학습(Transfer learning)의 적용이 필수적이다.

### 3. 모델 구조의 병렬화 (RNN vs Transformer)

LSTM과 같은 RNN 기반 모델은 시퀀스 데이터의 시간적 의존성을 잘 포착하지만 순차적 처리 방식으로 인해 병렬 연산이 불가능하다. 반면, Transformer 기반 모델은 전체 시퀀스를 한 번에 처리하여 연산 속도를 높일 수 있으나, Self-attention 메커니즘으로 인해 긴 시퀀스 처리 시 메모리와 계산 비용이 급격히 증가하는 한계가 있다.

### 4. 프라이버시와 설명 가능성

차량 데이터의 민감성을 해결하기 위해 데이터를 중앙 서버로 모으지 않고 로컬에서 학습하는 **Federated Learning**이 유망한 대안으로 제시된다. 또한, AI의 판단 근거를 제공하는 **Explainability (xNN 등)** 연구는 실제 차량 시스템에 AI를 적용하기 위한 신뢰성 확보 차원에서 매우 중요하다.

## 📌 TL;DR

본 논문은 CAN 버스 기반 차량 내 네트워크의 이상 탐지 기법을 통계적 방법부터 최신 딥러닝까지 망라하여 분석한 종합 서베이 보고서이다. 특히 학습 패러다임별(지도, 준지도, 비지도) 특성과 함께 다양한 공개 데이터셋의 성능을 비교 분석하였으며, 향후 연구 방향으로 모델의 경량화, 병렬 처리, 데이터 프라이버시 보호 및 설명 가능한 AI(XAI)의 도입을 제안한다. 이 연구는 차량 보안 및 유지보수 시스템을 설계하는 연구자들에게 최적의 알고리즘과 데이터셋을 선택하는 데 중요한 기준을 제공할 것으로 기대된다.
