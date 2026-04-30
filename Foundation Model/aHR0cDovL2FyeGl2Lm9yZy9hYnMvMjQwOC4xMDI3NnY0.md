# FEDKIM: Adaptive Federated Knowledge Injection into Medical Foundation Models

Xiaochen Wang, Jiaqi Wang, Houping Xiao, Jinghui Chen, Fenglong Ma (2024)

## 🧩 Problem to Solve

현대 인공지능 분야에서 Foundation Model은 다양한 모달리티와 태스크를 처리하는 뛰어난 능력을 보여주었으나, 의료 분야에서의 적용은 두 가지 핵심적인 제약 사항으로 인해 어려움을 겪고 있다. 첫째는 데이터 프라이버시 문제이다. 의료 데이터는 매우 민감하며, 미국의 HIPAA나 유럽의 GDPR과 같은 엄격한 법적 규제로 인해 데이터를 한곳에 모아 대규모 중앙 집중식 학습(Centralized Training)을 진행하는 것이 현실적으로 불가능하다. 둘째는 모달리티 및 태스크 적응성의 한계이다. 기존의 의료 Foundation Model들은 특정 모달리티나 좁은 범위의 하위 태스크에 특화되어 있어, 실제 임상 현장에서 요구되는 다중 모달리티 통합 분석 및 복잡한 의료 의사결정 능력이 부족하다.

본 논문의 목표는 이러한 프라이버시 제약을 준수하면서도, 분산된 의료 데이터로부터 지식을 효과적으로 추출하여 기존의 의료 Foundation Model에 주입(Injection)함으로써 모델의 규모를 확장하고 다중 모달리티 및 다중 태스크 처리 능력을 향상시키는 FEDKIM 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 의료 Foundation Model을 서버에 배치하고, 클라이언트(병원 등)에서는 경량화된 로컬 모델을 통해 지식을 추출한 뒤 이를 서버의 모델로 전송하여 통합하는 '연합 지식 주입(Federated Knowledge Injection)' 방식이다. 특히, 주입된 지식을 효율적으로 활용하기 위해 **Adaptive Multitask Multimodal Mixture Of Experts ($M^3OE$)** 모듈을 설계하였다. 이 모듈은 주어진 태스크의 설명과 모달리티 정보를 기반으로 최적의 전문가 시스템(Expert System)을 동적으로 선택하여, 복잡한 의료 맥락에서도 높은 적응력을 갖도록 설계되었다.

## 📎 Related Works

기존의 연합 학습(Federated Learning, FL) 기반 Foundation Model 연구들은 주로 기존 모델을 활용해 로컬 클라이언트에게 최적화된 서비스를 제공하는 것에 집중하였다. 그러나 분산된 환경에서 새로운 의료 지식을 기존 Foundation Model에 주입하는 문제는 충분히 다루어지지 않았다. 또한, Low-Rank Adaptation (LoRA)나 Mixture of Experts (MoE)와 같은 파라미터 효율적 미세 조정(PEFT) 기법들이 제안되었으나, 의료 분야의 특수한 다중 모달리티-다중 태스크 시나리오를 해결하기 위한 구조적 설계는 미흡한 실정이다. FEDKIM은 이러한 한계를 극복하기 위해 MoE 구조에 태스크 및 모달리티 인식을 결합하여 일반적인 PEFT보다 뛰어난 일반화 성능을 달성하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
FEDKIM은 로컬 클라이언트에서 동작하는 **Knowledge Extractor**와 서버에서 동작하는 **Knowledge Injector**로 구성된다.

### 2. 클라이언트 업데이트 (Knowledge Extraction)
각 클라이언트는 $M$개의 모달리티별 인코더($\text{ENC}_{n,m}$)와 $T$개의 태스크별 디코더($\text{DEC}_{n,t}$)를 가진 경량 모델 $f_n$을 학습시킨다. 학습 목표는 다음과 같은 손실 함수를 최소화하는 것이다.

$$\min_{\theta^n} L^n := \frac{1}{T} \sum_{t=1}^{T} \frac{1}{|D^n_t|} \sum_{(x^t_i, y^t_i) \in D^n_t} \ell_t(f_n(x^t_i; \theta^n), y^t_i)$$

여기서 $f_n(x_i; \theta^n) = \text{DEC}_{n,t}(\text{ENC}_{n,m}(x_i; \theta^{\text{enc}}_{n,m}); \theta^{\text{dec}}_{n,t})$이며, 학습이 완료된 인코더와 디코더의 파라미터 $\theta^{\text{enc}}_n, \theta^{\text{dec}}_n$는 서버로 전송된다.

### 3. 서버 업데이트 (Knowledge Injection)
서버는 전송받은 파라미터들을 FedAvg 또는 FedProx와 같은 연합 학습 알고리즘을 통해 집계하여 글로벌 인코더 $\theta_e$와 디코더 $\theta_d$를 생성한다. 이후 다음의 3단계 과정을 통해 Foundation Model $F$에 지식을 주입한다.

**Step 1: Feature Alignment**
입력 데이터 $x^t_j$에 대해 집계된 인코더 $\theta_e$를 사용하여 특징 표현 $e^t_j$를 추출하고, 이를 선형 매핑 함수 $g(\cdot)$를 통해 변환한다. 이 특징값과 Foundation Model $F$의 텍스트 임베딩 층을 통해 얻은 태스크 프롬프트 특징 $p_t$를 결합하여 최종 입력 $h^t_j = [e^t_j; p_t]$를 생성한다.

**Step 2: Multimodal Multi-tasking Mixture of Experts ($M^3OE$)**
$M^3OE$ 모듈은 태스크 설명 $T_t$와 모달리티 설명 $M_t$를 입력받아 전문가 선택 가중치 $\alpha_t$를 계산한다.

$$\beta_t = \frac{(W^q \text{EMB}_F(M_t))(W^k \text{EMB}_F(T_t))^\top}{\sqrt{d_k}} W^v \text{EMB}_F(T_t)$$
$$\alpha_t = \text{softmax}(\text{MLP}(\text{Pooling}(\beta_t)))$$

이 과정에서 $\alpha_t \in \mathbb{R}^P$는 $P$개의 전문가 시스템 중 어떤 전문가를 사용할지를 결정하는 가중치가 된다.

**Step 3: LoRA-$M^3OE$ 기반 PEFT**
최종적으로 LoRA 구조를 활용하여 $F$의 각 레이어 표현 $c^t_j$를 다음과 같이 계산한다.

$$c^t_j = W_F h^t_j + \sum_{p=1}^{P} \alpha^t_p (B_p A_p h^t_j)$$

여기서 $W_F$는 고정된(frozen) 파라미터이며, $B_p A_p$는 $p$번째 전문가 시스템에 해당하는 저차원 적응 모듈(LoRA)이다. 이를 통해 모델은 적은 파라미터 업데이트만으로도 효율적으로 지식을 흡수할 수 있다.

## 📊 Results

### 실험 설정
- **데이터셋 및 태스크**: 6개 모달리티에 걸친 4개의 학습 태스크(COVID-19 탐지, 폐 혼탁 탐지, ECG 이상 탐지, 사망률 예측)와 7개 모달리티에 걸친 8개의 검증 태스크(Zero-shot 평가용)를 사용하였다.
- **기준선(Baselines)**: 단순히 인코더만 통합하는 $\text{FedPlug}$, 여기에 LoRA를 추가한 $\text{FedPlug}_L$과 비교하였다.
- **백본 모델**: 의료 전용 LLM인 $\text{MMedLM-2}$ (7B 파라미터)를 사용하였다.

### 주요 결과
1. **Zero-shot 성능**: FEDKIM은 학습 시 보지 못한(unseen) 태스크에 대해 $\text{FedPlug}$ 및 $\text{FedPlug}_L$보다 월등한 성능을 보였다. 특히 Signal Noise Clarification (SNC) 태스크에서 FedAvg 기반 적용 시 $\text{FedPlug}_L$ 대비 82.36%의 성능 향상을 기록하였다.
2. **Fine-tuning 성능**: 학습에 사용된 친숙한 태스크들에 대해서도 FEDKIM은 모든 지표(Accuracy, Precision, Recall, F1)에서 기준선 모델들을 압도하였다.
3. **연합 알고리즘 영향**: FedAvg보다 FedProx를 백본으로 사용했을 때 전반적으로 더 높은 성능을 보였으며, 이는 지식 추출 과정에서의 정규화가 중요함을 시사한다.
4. **Ablation Study**: 공공 데이터만 사용한 $\text{FEDKIM}_{\text{pub}}$의 성능이 크게 하락한 것을 통해, 로컬 클라이언트로부터 주입된 프라이빗 지식의 중요성이 입증되었다. 또한 태스크 설명($T$)이나 모달리티 설명($M$) 모듈을 제거했을 때 성능이 저하되어 $M^3OE$의 설계 타당성이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 의료 데이터의 프라이버시 문제를 해결하면서도 Foundation Model의 기능을 확장할 수 있는 실질적인 프레임워크를 제시하였다. 특히 $M^3OE$ 모듈이 단순한 파라미터 통합을 넘어, 컨텍스트(태스크 및 모달리티 정보)에 따라 전문가를 선택적으로 활용하게 함으로써 zero-shot 일반화 능력을 크게 향상시킨 점이 인상적이다.

다만, 연구의 한계점으로 현재 계산 자원의 제약으로 인해 PEFT(LoRA) 방식만을 사용하였으며, 전체 파라미터 미세 조정(Full-parameter Fine-tuning)과의 비교가 이루어지지 않았다. 또한 7B 규모의 모델로 실험을 진행하였으나, 더 거대한 규모의 Foundation Model에서의 확장성(Scalability)에 대해서는 추가적인 검증이 필요하다.

## 📌 TL;DR

FEDKIM은 프라이버시 보호를 위해 분산된 의료 데이터에서 지식을 추출하고, 이를 $\text{M}^3\text{OE}$(Multimodal Multi-tasking Mixture of Experts) 모듈과 LoRA를 통해 의료 Foundation Model에 주입하는 프레임워크이다. 실험 결과, 이 방식은 기존의 단순 통합 방식보다 다중 모달리티 처리 능력이 뛰어나며, 특히 학습하지 않은 새로운 의료 태스크에 대해서도 강력한 Zero-shot 성능을 보여준다는 점에서 향후 프라이버시 보존형 의료 AI 모델 구축에 중요한 기여를 할 것으로 평가된다.