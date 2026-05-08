# Personalizing Keyword Spotting with Speaker Information

Beltrán Labrador, Pai Zhu, Guanlong Zhao, Angelo Scorza Scarpati, Quan Wang, Alicia Lozano-Diez, Alex Park, Ignacio López Moreno (2023)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 Keyword Spotting (KWS) 시스템이 다양한 억양(accents)과 연령층을 가진 사용자 집단에 대해 일반화 성능이 떨어진다는 점이다. 일반적으로 모델의 용량을 키우고 방대한 데이터를 학습시켜 이 문제를 해결할 수 있으나, KWS는 주로 메모리, 연산량, 전력 소모에 제한이 있는 저전력 IoT 기기(Low-resource devices)에서 동작해야 하므로 단순한 모델 크기 증가는 현실적인 해결책이 되지 않는다.

따라서 본 논문의 목표는 모델의 파라미터 수와 연산 비용의 증가를 최소화하면서도, 화자 정보(Speaker information)를 활용하여 특정 사용자에게 개인화된 KWS 시스템을 구축함으로써 다양한 사용자 환경에서의 인식 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Feature-wise Linear Modulation (FiLM) 기법을 도입하여 화자 임베딩(Speaker embedding)을 KWS 모델에 통합하는 것이다. FiLM은 신경망의 중간 레이어 출력값에 아핀 변환(Affine transformation)을 적용함으로써, 입력된 화자의 고유한 특성에 따라 모델의 표현(Representation)을 동적으로 조정할 수 있게 한다. 이를 통해 모델의 크기를 거의 늘리지 않고도 화자별 특성에 맞게 적응(Adaptation)하는 효과를 얻을 수 있다.

## 📎 Related Works

기존의 화자 적응 방식으로는 ASR(Automatic Speech Recognition) 분야의 fMLLR 특징이나 화자 임베딩 기반의 조건화 방식이 사용되었다. KWS 분야에서도 VoiceFilter를 이용한 음성 추출 기반의 전처리 방식이나, 키워드 검출과 화자 식별을 동시에 학습하는 Multi-task learning 접근법이 제안된 바 있다.

하지만 Multi-task learning 방식은 두 가지 작업을 동시에 수행함에 따라 연산 복잡도가 증가하고 모델의 해석 가능성이 떨어진다는 한계가 있다. 본 연구는 이러한 복잡성을 피하기 위해, 사전 학습된 화자 인식 시스템에서 추출한 임베딩을 FiLM을 통해 주입하는 방식을 채택하여 효율성과 실용성을 동시에 확보하고자 하였다.

## 🛠️ Methodology

### 1. Baseline KWS System

베이스라인 모델은 저리소스 환경에 최적화된 엔드투엔드(End-to-end) 신경망 구조를 사용한다.

- **구조**: Encoder-Decoder 아키텍처이며, 총 파라미터 수는 $350\text{K}$개이다.
- **Encoder**: 4개의 Singular Value Decomposition Filter (SVDF) 레이어(각 576 노드)와 64 크기의 Bottleneck 레이어로 구성된다.
- **Decoder**: 3개의 SVDF 레이어(각 32 노드)로 구성된다.
- **학습 목표**: 특징 시퀀스 $X$와 라벨 시퀀스 $Y$에 대해 Cross-Entropy (CE) 손실 함수를 최소화하도록 학습한다.
$$\theta_{base} = \operatorname{argmin}_{\theta} \mathbb{E}_{(x,y)} [L_{CE}(f(x; \theta), y)]$$

### 2. Speaker Embedding Extraction

두 가지 종류의 화자 인식 시스템을 통해 임베딩을 추출한다.

- **Text-Dependent (TD)**: 특정 문구("Okay/Hey Google")를 말했을 때만 추출 가능하며, 3개의 LSTM 레이어를 통해 64차원 임베딩을 생성한다. 파라미터 수는 $235\text{k}$개로 매우 가볍다.
- **Text-Independent (TI)**: 자유로운 발화에서도 추출 가능하며, Conformer 기반의 인코더(12개 레이어, 각 256차원)와 Attentive temporal pooling을 사용하여 256차원 임베딩을 생성한다. 총 파라미터 수는 22M개이다.

### 3. Personalization via FiLM

FiLM 레이어를 Encoder와 Decoder 사이의 출력 로짓(Logits)에 적용하여 화자 정보를 주입한다. FiLM의 동작 방식은 다음과 같이 단순한 선형 변환으로 정의된다.
$$\text{FiLM}(l, \gamma, \beta) = \gamma \odot l + \beta$$
여기서 $l$은 레이어의 출력값이며, $\gamma$ (scaling)와 $\beta$ (bias)는 화자 임베딩으로부터 학습 가능한 투영 레이어(Projection layers)를 통해 생성된 값이다. $\odot$은 요소별 곱셈(Element-wise multiplication)을 의미한다.

최종적인 학습 목적 함수는 다음과 같다.
$$\theta_{cond} = \operatorname{argmin}_{\theta} \mathbb{E}_{(x,y,s)} [L_{CE}(f(x, s; \theta), y)]$$
여기서 $s$는 등록된 화자 임베딩 집합 $S$에 속하는 화자 정보를 의미한다.

### 4. Robust Training Strategy

실제 환경에서는 화자 등록(Enrollment) 단계가 생략되거나 실패할 수 있다. 이를 위해 화자 임베딩이 있는 데이터와 없는 데이터(임베딩 벡터를 동일한 차원의 상수 벡터로 대체)를 섞어서 학습시키는 Robust training을 수행하여, 임베딩 유무와 상관없이 키워드 검출이 가능하도록 설계하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: 미국, 인도, 영국, 호주 등 다양한 영어 억양과 전 연령층(특히 12세 미만 아동 포함)을 포함한 벤더 제공 데이터를 사용하였다.
- **평가 지표**: Equal Error Rate (EER) 및 DET (Detection Error Tradeoff) 곡선을 사용하였다.
- **비교 방법론**:
    1. **TI Self-Enrollment**: 입력 음성 자체에서 TI 임베딩을 추출하여 조건화.
    2. **TI Cross-Enrollment**: 별도의 등록 음성에서 TI 임베딩을 추출하여 조건화.
    3. **TD Cross-Enrollment**: 별도의 등록 음성(키워드 구간)에서 TD 임베딩을 추출하여 조건화.

### 주요 결과

- **TI Self-Enrollment**: 베이스라인 대비 EER 기준 $18.7\%$의 상대적 개선을 보이며 가장 높은 성능을 기록하였다. 하지만 추론 시 매번 임베딩을 추출해야 하므로 지연 시간(Latency)과 연산 비용이 증가한다.
- **TI Cross-Enrollment**: 오히려 성능이 저하되는 경향을 보였다. 이는 256차원의 고차원 임베딩이 단일 레이어 FiLM에 과적합(Underfit)되었거나, 일반 발화 기반 임베딩이 화자 특성을 충분히 캡처하지 못했기 때문으로 분석된다.
- **TD Cross-Enrollment**: 베이스라인 대비 $2.6\%$의 EER 개선을 보였으며, 특히 **인도 억양의 아동 데이터에서 $24\%$라는 괄목할 만한 개선**을 보였다. 임베딩이 사전 계산되어 저장되므로 추론 시 추가 연산 비용이 거의 없다는 실용적 장점이 있다.
- **Robust Training**: 화자 임베딩이 없는 경우 TD 조건화 모델은 완전히 실패(Failure)하지만, Robust training을 적용한 모델은 베이스라인 수준의 성능을 유지하면서 임베딩이 있을 때는 성능 향상을 그대로 누리는 결과를 보였다.

## 🧠 Insights & Discussion

본 연구는 매우 적은 파라미터 증가(약 $1\%$)만으로도 KWS 시스템의 개인화를 달성할 수 있음을 보여주었다. 특히 주목할 점은 데이터셋에서 상대적으로 소외된 그룹(Underrepresented groups), 즉 특정 국가의 억양을 가진 아동 집단에서 성능 향상 폭이 매우 컸다는 것이다. 이는 AI Fairness 관점에서 기술의 포용성을 높이는 데 기여할 수 있음을 시사한다.

다만, TI 방식의 Cross-Enrollment가 기대보다 낮은 성능을 보인 점은 FiLM 레이어의 단순성이나 임베딩 차원수와 모델 용량 간의 불균형 문제일 가능성이 크다. 또한, TD 방식은 반드시 키워드 발화 등록 과정이 필요하다는 제약이 있다. 그럼에도 불구하고, 실시간 처리 제약이 심한 온디바이스 환경에서는 사전 계산된 TD 임베딩을 활용하는 방식이 가장 합리적인 대안임을 입증하였다.

## 📌 TL;DR

본 논문은 **FiLM(Feature-wise Linear Modulation)** 기법을 사용하여 화자 임베딩을 가벼운 KWS 모델에 통합함으로써, 연산 비용 증가를 최소화하면서 개인화된 키워드 검출을 가능하게 하였다. 특히 **Text-Dependent(TD) 임베딩 기반의 개인화**가 실용성 측면에서 우수하며, 억양이 강한 아동 등 특정 사용자 그룹의 인식률을 크게 개선함으로써 서비스의 포용성을 높일 수 있음을 확인하였다. 이 연구는 향후 저전력 온디바이스 음성 인터페이스의 맞춤형 성능 향상을 위한 효율적인 프레임워크를 제공한다.
