# SPEECH-MAMBA: LONG-CONTEXT SPEECH RECOGNITION WITH SELECTIVE STATE SPACES MODELS

Xiaoxue Gao and Nancy F. Chen (2024)

## 🧩 Problem to Solve

현재의 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템은 주로 Transformer 기반 모델을 사용하고 있으나, 입력 시퀀스 길이에 따라 계산 복잡도가 제곱으로 증가하는 $\mathcal{O}(L^2)$의 Quadratic Complexity 문제를 가지고 있다. 이로 인해 매우 긴 음성 시퀀스를 모델링할 때 연산 비용이 급격히 증가하며, 결과적으로 장거리 의존성(Long-range dependencies)을 효과적으로 캡처하는 데 한계가 발생한다.

본 논문의 목표는 Selective State Space Models(SSMs)인 Mamba를 Transformer 아키텍처에 통합하여, 연산 복잡도를 선형 수준(Near-linear complexity)으로 낮추면서도 긴 컨텍스트의 음성 데이터를 정확하게 인식할 수 있는 Speech-Mamba 모델을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Transformer의 국소적/시간적 표현 능력과 Mamba의 전역적/장거리 컨텍스트 모델링 능력을 결합하는 하이브리드 구조를 설계하는 것이다. Transformer는 낮은 수준의 음성 및 텍스트 표현(Lower-level representation)을 모델링하는 데 집중하고, Mamba는 선택적 상태 공간 모델을 통해 긴 시퀀스에서 중요한 정보를 선택적으로 유지하며 깊은 수준의 표현(Deeper representation)을 학습하도록 설계되었다.

## 📎 Related Works

기존의 End-to-End(E2E) ASR 모델은 Connectionist Temporal Classification(CTC)과 Sequence-to-Sequence(S2S) 모델, 또는 이 둘을 결합한 Joint CTC-S2S 구조를 주로 사용해 왔다. 특히 Transformer 아키텍처는 어텐션 메커니즘을 통해 시퀀스 변환 능력을 획기적으로 향상시켰으나, 앞서 언급한 복잡도 문제로 인해 아주 긴 시퀀스 처리에는 부적합하다는 한계가 있다.

최근 NLP 및 CV 분야에서는 S4(Structured State Space Sequence Models)를 계승한 Mamba가 등장하였다. Mamba는 입력 데이터에 따라 관련 정보를 선택적으로 수용하는 Selective Mechanism을 도입하여, Transformer 수준의 성능을 유지하면서도 선형적인 계산 복잡도를 달성하였다. 본 논문은 이러한 Mamba의 이점을 ASR 분야에 적용하여 기존 Transformer 기반 모델의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
Speech-Mamba는 Joint Encoder-Decoder 프레임워크를 기반으로 하며, CTC loss를 함께 사용하여 음성 입력을 텍스트 출력으로 변환한다.

*   **Mamba Encoder**: 음성 입력의 어쿠스틱 특징(Acoustic features)을 받아 전역적인 장거리 컨텍스트가 반영된 중간 표현으로 변환한다. $M$개의 Mamba Encoder Block으로 구성된다.
*   **Mamba Decoder**: 텍스트 임베딩과 인코더의 출력을 받아 자동 회귀(Auto-regressive) 방식으로 텍스트 시퀀스를 예측한다. $N$개의 Mamba Decoder Block으로 구성된다.

### 2. 주요 구성 요소 및 블록 설계
각 블록은 단순한 Mamba 층의 반복이 아니라, Transformer의 어텐션 메커니즘과 Mamba의 SSM을 상호 보완적으로 배치한 구조를 가진다.

*   **Mamba Encoder Block**: $\text{Mamba Block} \rightarrow \text{RMSNorm} \rightarrow \text{Multi-head Attention (RMS-ATT)} \rightarrow \text{Mamba Block}$ 순으로 구성된다.
*   **Mamba Decoder Block**: $\text{Mamba Block} \rightarrow \text{RMSNorm} \rightarrow \text{Source-Target Attention (RMS-STA)} \rightarrow \text{Mamba Block}$ 순으로 구성된다.
*   **Mamba Block 내부 구조**: 
    1. 입력 데이터는 먼저 $\text{RMSNorm}$을 거친다.
    2. $\text{MLP}$ 및 $\text{Gated MLP}$를 통해 특징이 확장된다.
    3. 이후 $\text{1D Convolution} \rightarrow \text{SiLU Activation} \rightarrow \text{Selective SSM}$ 과정을 거치며 핵심 컨텍스트 정보가 압축된다.
    4. 마지막으로 $\text{Residual Connection}$과 $\text{Dropout}$이 적용된다.

여기서 $\text{RMSNorm}$은 레이어 활성화 값의 크기를 안정화하여 학습의 안정성을 높이는 역할을 한다.

### 3. 학습 목표 및 손실 함수
모델은 CTC loss와 S2S loss를 동시에 최소화하는 Multi-objective Learning 방식을 사용한다. 전체 손실 함수는 다음과 같이 정의된다.

$$L_{\text{Speech-Mamba}} = \alpha L_{\text{CTC}} + (1-\alpha) L_{\text{S2S}}$$

여기서 $\alpha$는 CTC loss의 가중치이며, 본 연구에서는 $\alpha=0.3$으로 설정하였다. $L_{\text{CTC}}$는 인코더 출력과 타겟 텍스트 간의 단조적 정렬(Monotonic alignment)을 보장하며, $L_{\text{S2S}}$는 디코더가 예측한 텍스트와 타겟 텍스트 간의 Cross-entropy loss이다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: LibriSpeech 데이터셋을 사용하였다. 특히 장거리 컨텍스트 성능 평가를 위해, 동일 화자의 발화들을 연결하여 $45$초에서 $100$초 사이의 길이를 가진 별도의 Long-context 데이터셋(dev-clean-L, test-clean-L 등)을 직접 구축하였다.
*   **기준선(Baseline)**: Joint CTC-S2S Transformer 모델, Trans-CTC, Mamba-CTC 모델과 비교하였다.
*   **측정 지표**: Word Error Rate (WER)를 사용하였다.

### 2. 주요 결과
*   **일반 인식 성능**: 표준 테스트 세트에서 Speech-Mamba는 Transformer 베이스라인보다 우수한 성능을 보였다.
*   **장거리 시퀀스 모델링**: 특히 긴 음성 데이터에서 압도적인 성능 향상이 나타났다. Transformer 대비 WER이 상대적으로 $65\%$ 이상 개선되었으며, 특정 세트(test-clean-L, dev-clean-L)에서는 최대 $84\%$의 개선율을 보였다.
*   **SOTA 모델 비교**: 960시간의 데이터로 학습된 Speech-Mamba(파라미터 수 67.6M)는 훨씬 더 많은 파라미터를 가진 Whisper-Large-V3(1550M)보다 모든 테스트 세트에서 더 낮은 WER을 기록하였다. 또한, 매우 긴 컨텍스트 처리 능력을 가진 Gemini-1.5-Pro와 비교했을 때도 clean-L 세트에서 더 우수한 성능을 보였다.

### 3. Ablation Study
*   **Mamba Encoder 제거**: 긴 시퀀스에서 성능이 급격히 저하되어, 전역 음성 표현 학습에 Mamba 인코더가 필수적임이 확인되었다.
*   **Mamba Decoder 제거**: 성능 저하가 나타났으나 인코더 제거보다는 덜 치명적이었다. 이는 Mamba 디코더가 텍스트-음성 간의 교차 모달 관계를 유지하는 데 기여함을 시사한다.
*   **Multi-objective 제거**: S2S loss를 제거하고 CTC loss만 사용했을 때 전반적인 성능이 하락하여, 두 손실 함수를 함께 사용하는 것이 효과적임이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 Mamba의 선형 복잡도라는 이론적 강점이 실제 ASR의 장거리 컨텍스트 문제 해결에 매우 실용적인 도구가 될 수 있음을 실험적으로 증명하였다. 특히 Transformer가 시퀀스 길이가 길어짐에 따라 WER이 급격히 증가하는 반면, Speech-Mamba는 $100$초에 이르는 긴 발화에서도 안정적인 성능을 유지한다는 점이 고무적이다.

또한, 파라미터 효율성 측면에서 Whisper-Large-V3와 같은 거대 모델보다 훨씬 적은 파라미터(약 $1/23$ 수준)로도 더 나은 성능을 낼 수 있다는 점은, 적절한 아키텍처 설계가 단순한 모델 크기 확장보다 더 중요할 수 있음을 시사한다. 

다만, 본 연구는 주로 영어 데이터셋인 LibriSpeech에 집중되어 있으며, 다양한 언어나 극심한 노이즈가 포함된 실제 환경에서의 강건성(Robustness)에 대한 분석은 명시되지 않았다. 향후 다른 언어 및 다양한 음성 처리 태스크로의 확장이 필요할 것으로 보인다.

## 📌 TL;DR

이 논문은 Transformer의 $\mathcal{O}(L^2)$ 복잡도 문제를 해결하기 위해 Selective SSM인 Mamba를 통합한 **Speech-Mamba**를 제안한다. Mamba의 선형 계산 복잡도와 Transformer의 시간적 표현 능력을 결합하여, 특히 $45\text{--}100$초 길이의 긴 음성 시퀀스 인식에서 기존 Transformer 및 SOTA 모델(Whisper-Large-V3 등)보다 훨씬 적은 파라미터로 더 높은 정확도(낮은 WER)를 달성하였다. 이는 차세대 장거리 컨텍스트 음성 인식 시스템 구축을 위한 효율적인 기반 모델이 될 가능성이 높다.