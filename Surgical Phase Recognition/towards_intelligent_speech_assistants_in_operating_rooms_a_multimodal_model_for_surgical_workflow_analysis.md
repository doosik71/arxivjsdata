# Towards Intelligent Speech Assistants in Operating Rooms: A Multimodal Model for Surgical Workflow Analysis

Kubilay Can Demir, Belén Lojo Rodríguez, Tobias Weise, Andreas Maier, Seung Hee Yang (2024)

## 🧩 Problem to Solve

수술실(Operating Room, OR) 내의 수술 과정은 점차 복잡해지고 있으며, 이에 따라 의료진의 책임과 인원수가 계속해서 증가하고 있다. 이러한 환경에서 루틴한 작업을 대신 수행하거나 의료진을 보조할 수 있는 지능형 음성 비서(Intelligent Speech Assistant)의 필요성이 대두되었다. 이러한 시스템이 실질적으로 작동하기 위해서는 현재 수술이 어느 단계에 와 있는지를 정확하게 파악하는 수술 워크플로우 분석(Surgical Workflow Analysis, SWA), 그 중에서도 수술 단계 인식(Surgical Phase Recognition)이 필수적인 전제 조건이다.

본 논문의 목표는 포트 카테터 삽입술(Port-catheter placement operations)을 대상으로, 음성(Speech)과 이미지(Image) 데이터를 결합한 멀티모달 프레임워크를 구축하여 수술 단계 인식의 정확도와 효율성을 높이는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 모든 수술 단계에 하나의 모델을 적용하는 대신, 수술 단계의 특성에 따라 서로 다른 데이터 모달리티가 더 중요한 역할을 한다는 점에 착안하여 **음성 기반 모델과 이미지 기반 모델을 분리하여 설계하고 이를 적절한 시점에 병합(Merging)**하는 것이다.

구체적으로, 수술 초기 단계인 준비(Preparation)와 천자(Puncture) 단계에서는 음성 데이터가 결정적인 단서를 제공하며, 이후의 가이드 와이어 배치(Positioning of the Guide Wire)부터 종료(Closing) 단계까지는 X-ray 이미지와 장비 로그 데이터가 더 중요한 정보를 제공한다. 이를 통해 복잡한 전체 문제를 더 작고 단순한 문제들로 나누어 해결함으로써 인식 성능을 향상시켰다.

## 📎 Related Works

기존의 수술 단계 인식 연구들은 주로 복강경 담낭 절제술(Laparoscopic Cholecystectomy)이나 백내장 수술(Cataract Surgery)과 같은 사례에 집중되었으며, 주로 내시경 비디오나 현미경 비디오를 주 데이터 소스로 사용하였다. 일부 연구에서는 음성이나 오디오 데이터를 활용하거나, X-ray 이미지와 의료진의 음성 채널을 결합한 시도가 있었으나(예: PoCaPNet), 본 논문은 여기서 더 나아가 다음과 같은 차별점을 가진다.

1. **모달리티의 세분화 및 보완**: 단순한 음성/이미지 결합을 넘어, X-ray 기기의 로그 파일(Log files) 데이터를 추가하여 이미지 모델의 성능을 보완하였다.
2. **단계별 모델 전환 전략**: 모든 단계를 하나의 모델로 예측하는 것이 아니라, 수술 흐름에 따라 음성 모델에서 이미지 모델로 전환하는 전략을 사용하여 각 모달리티의 효율성을 극대화하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 특징 추출 (Feature Extraction)

시스템은 음성 모델과 이미지 모델이라는 두 개의 핵심 경로로 구성된다.

* **음성 모델(Speech Model)**: 세 가지 채널을 사용한다.
  * **Physician & Assistant 채널**: 깨끗한 음성 신호가 기록되므로, 다국어 사전 학습 모델인 $\text{wav2vec 2.0 XLSR-53}$를 사용하여 특징을 추출한다.
  * **Ambient 채널**: 소음과 잔향이 많으므로, 배경 소음 및 비음성 신호를 더 잘 표현하기 위해 40차원의 $\text{Mel Frequency Cepstral Coefficients (MFCC)}$를 추출한다.
* **이미지 모델(Image Model)**:
  * **X-ray 이미지**: $\text{TorchXRayVision}$ 라이브러리의 $\text{Densenet121}$ 모델(흉부 X-ray 데이터로 사전 학습됨)을 사용하여 특징을 추출한다.
  * **X-ray 로그**: X-ray 기기의 동작 상태(Fluoroscopy 모드 여부, DSA 모드 여부, 기기 이동 여부)를 나타내는 3가지 변수를 원-핫 인코딩(One-hot encoding)하고, 이를 64번 반복하여 192차원의 특징 벡터로 만들어 이미지 특징과 결합한다.

### 2. Gated Multimodal Units (GMU)

음성 모델에서는 서로 다른 세 가지 오디오 채널의 특징을 효과적으로 융합하기 위해 GMU를 사용한다. 각 모달리티 $k$에 대해 다음과 같이 계산된다.

$$h_k = \tanh(W_{hk} \cdot x_k)$$
$$z_k = \sigma(W_{zk} \cdot [x_1, \dots, x_K])$$
$$o_k = h_k \cdot z_k$$

여기서 $x_k$는 입력 특징 벡터이며, $W_{hk}$와 $W_{zk}$는 학습 가능한 가중치 행렬이다. $z_k$는 각 모달리티의 중요도를 결정하는 게이트 역할을 하며, 최종 출력 $h$는 모든 모달리티의 출력 벡터 $o_k$의 합으로 계산된다.
$$h = \sum_{k=1}^{K} o_k$$

### 3. Multi-Stage Temporal Convolutional Network (MS-TCN)

시간적 관계를 모델링하기 위해 MS-TCN 아키텍처를 사용한다. 이는 여러 단계의 $\text{Single-Stage TCN}$이 쌓인 구조로, $\text{Dilated Convolution}$을 통해 수용 영역(Receptive Field)을 넓힌다. 잔차 연결(Residual connection)이 포함된 Dilated Residual Convolutional Layer의 수식은 다음과 같다.

$$\hat{d}_l = \text{ReLU}(W_{1,l} * d_{l-1} + b_{1,l})$$
$$d_l = d_{l-1} + W_{2,l} * \hat{d}_l + b_{2,l}$$

여기서 $*$는 컨볼루션 연산을 의미하며, 최종적으로 $\text{Softmax}$ 함수를 통해 각 시점 $t$에서의 수술 단계 확률을 추정한다. 또한, 시계열 데이터의 연속성을 확보하기 위해 $\text{Autoregressive connection}$을 적용하였다.

### 4. 학습 절차 및 모델 병합 전략

* **손실 함수 및 최적화**: 클래스 불균형 문제를 해결하기 위해 $\text{Label-distribution-aware margin (LDAM)}$ 손실 함수를 사용하였으며, $\text{Adam}$ 옵티마이저를 적용하였다.
* **모델 전환 전략**:
  * **음성 모델**: Preparation $\rightarrow$ Puncture 단계 예측 담당.
  * **이미지 모델**: Positioning of the Guide Wire $\rightarrow$ Closing 단계 예측 담당.
  * **전환 시점**: 천자(Puncture) 단계에서 음성 모델이 일정 시간 동안 일관된 예측 결과를 출력하면 이미지 모델로 제어권을 전환한다.

## 📊 Results

### 실험 설정

* **데이터셋**: 비공개 데이터셋인 $\text{PoCaP Corpus}$를 사용하였으며, 총 28건의 수술 데이터를 훈련(18건), 검증(5건), 테스트(5건) 세트로 분할하였다.
* **지표**: 프레임 단위 정확도(Frame-wise Accuracy)와 매크로 평균 F1-Score(Macro-averaged F1-Score)를 측정하였다.

### 정량적 결과

제안 모델은 이전 연구인 $\text{PoCaPNet}$ 대비 약 10%의 성능 향상을 보였다.

| Model | Accuracy (%) | F1-Score (%) |
| :--- | :---: | :---: |
| PoCaPNet | $82.56 \pm 3.21$ | $81.30 \pm 3.89$ |
| **Ours** | $\mathbf{92.65 \pm 3.52}$ | $\mathbf{92.30 \pm 3.82}$ |

### 분석 및 고찰

1. **특정 단계 인식 개선**: 특히 이전 모델들이 인식하지 못했던 $\text{Catheter Positioning}$ 단계의 인식률이 크게 개선되었다. 다만, 과분할(Over-segmentation) 오류가 발생하는 한계가 관찰되었다.
2. **채널별 기여도 분석**:
    * **음성 모델**: 모든 채널을 사용했을 때 가장 성능이 좋았으며, 단일 채널 중에는 의사(Physician) 채널의 성능이 가장 높았다. 보조원(Assistant) 채널은 보조원이 수술실을 비우거나 다른 업무를 수행하는 경우가 많아 성능 저하가 뚜렷했다.
    * **이미지 모델**: X-ray 이미지 단독 사용 시보다 로그 파일 데이터를 추가했을 때 F1-Score가 약 15% 향상되어, 로그 데이터의 보완적 역할이 입증되었다.

## 🧠 Insights & Discussion

본 논문의 강점은 수술의 도메인 지식(Domain Knowledge)을 딥러닝 아키텍처에 직접적으로 반영했다는 점이다. 단순히 모든 데이터를 하나의 모델에 넣는 대신, 수술 단계별로 지배적인 모달리티가 다르다는 점을 이용하여 모델을 분리하고 전환하는 전략을 취함으로써 복잡도를 낮추고 정확도를 높였다.

또한, X-ray 기기의 로그 파일이라는 정형 데이터를 이미지 특징과 결합하여 이미지 모델의 취약점을 보완한 점이 인상적이다. 하지만 논문에서 언급되었듯이 $\text{Catheter Positioning}$ 단계에서 나타나는 과분할 문제는 여전히 해결해야 할 과제로 남아 있다. 이는 해당 단계의 지속 시간이 짧고 구현 방식의 변동성이 크기 때문으로 분석된다.

## 📌 TL;DR

본 연구는 포트 카테터 삽입술의 수술 단계 인식을 위해 **음성 기반 모델(wav2vec 2.0, MFCC, GMU)**과 **이미지 기반 모델(DenseNet121, X-ray logs, MS-TCN)**을 결합한 멀티모달 프레임워크를 제안하였다. 수술 흐름에 따라 모델을 전환하는 전략을 통해 이전 연구 대비 약 10% 향상된 성능(Accuracy 92.65%)을 달성하였으며, 특히 인식이 어려웠던 특정 단계의 검출 능력을 크게 개선하였다. 이 연구는 향후 수술실 내 지능형 음성 비서의 실시간 상황 인지 기능을 구현하는 데 중요한 기초가 될 것으로 보인다.
