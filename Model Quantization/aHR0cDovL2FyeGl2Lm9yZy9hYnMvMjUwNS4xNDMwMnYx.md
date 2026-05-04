# Scaling Law for Quantization-Aware Training

Mengzhao Chen, Chaoyi Zhang, Jing Liu, Yutao Zeng, Zeyue Xue, Zhiheng Liu, Yunshui Li, Jin Ma, Jie Huang, Xun Zhou, Ping Luo (2025)

## 🧩 Problem to Solve

거대 언어 모델(LLM)은 방대한 계산 리소스와 메모리를 요구하기 때문에 실제 배포 시 심각한 제약이 따른다. 이를 해결하기 위해 모델의 정밀도를 낮추는 양자화(Quantization) 기술이 사용되며, 특히 훈련 과정에서 양자화 오차를 반영하는 Quantization-Aware Training (QAT)은 Post-Training Quantization (PTQ)보다 더 공격적인 압축(예: 4-bit)에서도 성능 유지 능력이 뛰어나다.

그러나 4-bit 정밀도(W4A4)에서의 QAT Scaling behavior에 대한 이해는 여전히 부족한 상태이다. 기존의 QAT scaling law들은 주로 모델의 파라미터 수에만 집중했을 뿐, 훈련에 사용된 토큰의 수($D$)나 양자화의 세밀함 정도를 나타내는 Granularity($G$)가 성능에 미치는 영향을 간과하였다. 본 논문의 목표는 모델 크기, 훈련 데이터 양, 양자화 granularity를 모두 통합하여 양자화 오차를 예측할 수 있는 Unified Scaling Law를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 W4A4 QAT의 양자화 오차($\delta_p$)를 모델 크기($N$), 훈련 토큰 수($D$), 그리고 양자화 그룹 크기($G$)의 함수로 모델링한 통합 Scaling Law를 제시한 것이다.

단순히 수식을 제안하는 것에 그치지 않고, 268회의 대규모 QAT 실험을 통해 양자화 오차가 모델 크기가 커질수록 감소하지만, 훈련 데이터가 많아지거나 양자화 granularity가 거칠어질수록( coarser) 증가한다는 사실을 실증적으로 증명하였다. 또한, 양자화 오차를 가중치(Weight)와 활성화 함수(Activation) 성분으로 분해하여 분석함으로써, Feed-Forward Network의 FC2 레이어 입력단에서 발생하는 Outlier가 W4A4 QAT의 주요 병목 지점임을 밝혀내고 이를 해결하기 위한 Mixed-precision 전략의 효과를 입증하였다.

## 📎 Related Works

기존의 Scaling Law 연구는 Kaplan과 Chinchilla law를 통해 모델 크기, 데이터셋 크기, 계산량과 최종 Loss 사이의 관계를 정립하였다. 최근에는 이를 모델 압축 분야로 확장하여 PTQ 및 QAT에 대한 scaling law 연구가 진행되었다.

하지만 기존의 QAT scaling law들은 Effective Parameter Multiplier (EPM)라는 개념을 도입하여 모델 크기의 변화로만 성능 저하를 설명하려 했다. 즉, 양자화 오차가 훈련 데이터의 양($D$)이나 양자화 그룹 크기($G$)와는 독립적이라고 가정하는 한계가 있었다. 본 논문은 실제 실험을 통해 훈련 데이터가 증가할수록 BF16 모델과 QAT 모델 사이의 Loss 간격(양자화 오차)이 오히려 벌어진다는 점을 발견하였으며, 이를 수식에 명시적으로 포함함으로써 기존 연구들과의 차별성을 확보하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
본 연구는 Llama3 스타일의 모델 아키텍처를 기반으로 OLMo2-Mix-1124 데이터셋을 사용하여 훈련을 진행하였다. 모델 크기 $N \in \{74\text{M}, 145\text{M}, 297\text{M}, 595\text{M}\}$와 훈련 토큰 수 $D \in \{10\text{B}, 20\text{B}, 50\text{B}, 100\text{B}\}$의 조합으로 실험을 설계하였으며, 양자화 정밀도는 W4A4를 중심으로 분석하였다.

### Proposed QAT Scaling Law
본 논문은 최종 Loss를 Chinchilla loss와 저비트 QAT로 인해 발생하는 추가적인 양자화 오차($\delta_p$)의 합으로 정의한다.

$$L(N,D,G) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E + \delta_p(N,D,G)$$

여기서 $\delta_p(N,D,G)$는 다음과 같은 전력 법칙(Power-law) 형태로 모델링된다.

$$\delta_p(N,D,G) = \frac{k \cdot D^{\gamma_D} \cdot (\log_2(G))^{\gamma_G}}{N^{\gamma_N}}$$

각 변수의 역할은 다음과 같다:
- $N$: 모델 파라미터 수. $\gamma_N$은 모델 크기에 따른 오차 감소 민감도를 나타낸다.
- $D$: 훈련에 사용된 토큰 수. $\gamma_D$는 데이터 증가에 따른 오차 증가 민감도를 나타낸다.
- $G$: Quantization Group Size. $\log_2(G)$를 통해 granularity가 거칠어질수록 오차가 증가하는 경향을 반영하며, $\gamma_G$는 이에 대한 민감도를 나타낸다.
- $k$: 피팅된 상수.

### 훈련 및 양자화 설정
- **Quantizer**: 가중치 양자화에는 AbsMax를 사용하였으며, 활성화 함수 양자화에는 그룹 크기가 256 미만일 때는 AbsMax를, 256 이상일 때는 LAC(Learnable Activation Clipping)를 사용하여 Outlier 대응 능력을 높였다.
- **Precision**: Integer(INT4) 형식을 채택하였으며, 이는 FP4와 유사하거나 더 나은 성능을 보임을 확인하였다.
- **Evaluation Metric**: 검증 손실의 편향되지 않은 추정치로 smoothed training loss를 사용하였다.

## 📊 Results

### 정량적 분석 및 경향성
실험 결과, $\delta_{W4A4}$는 다음과 같은 뚜렷한 경향성을 보였다.
1. **모델 크기($N$)**: 모델이 커질수록 양자화 오차는 일관되게 감소한다. (74M $\rightarrow$ 594M 증가 시 평균 34% 감소)
2. **훈련 데이터($D$)**: 훈련 토큰 수가 많아질수록 양자화 오차는 증가한다. (10B $\rightarrow$ 100B 증가 시 평균 22% 증가)
3. **Granularity($G$)**: 그룹 크기가 작아질수록(더 세밀할수록) 오차가 감소한다.

### 오차 분해 (Weight vs Activation)
$\delta_{W4A4} \approx \delta_{W4A16} + \delta_{W16A4}$ 관계가 성립함을 확인하여, 전체 오차를 가중치 오차($\delta_{W4A16}$)와 활성화 오차($\delta_{W16A4}$)로 분해하여 분석하였다.
- **가중치 오차**: 모델 크기($N$)와 훈련 데이터($D$)의 변화에 훨씬 민감하게 반응한다.
- **활성화 오차**: 양자화 granularity($G$)의 변화에 매우 민감하며, 이는 활성화 값의 Outlier 때문이다.

### FC2Proj 레이어 병목 현상
Kurtosis(첨도) 분석을 통해 FC2Proj 레이어의 입력값이 다른 레이어보다 압도적으로 높은 Kurtosis를 가짐을 발견하였다. 이는 SwiGLU 모듈의 gating mechanism이 Outlier를 증폭시키기 때문이며, 결과적으로 W4A4 QAT의 주된 병목 지점이 된다. 이를 해결하기 위해 FC2Proj 입력만 8-bit로 유지하는 Mixed-precision을 적용했을 때, $G=256$ 환경에서 양자화 오차가 42.9%까지 감소하는 효과를 보였다.

## 🧠 Insights & Discussion

본 논문은 QAT에서 단순히 모델을 크게 만드는 것뿐만 아니라, 훈련 데이터의 양과 양자화 granularity 사이의 트레이드오프를 정밀하게 계산해야 함을 시사한다. 특히, 훈련 데이터를 무작정 늘리는 것이 항상 이득이 아니라, 오히려 BF16 베이스라인과의 간격(양자화 오차)을 벌릴 수 있다는 점은 매우 중요한 발견이다.

비판적 관점에서 볼 때, 본 연구는 Dense 모델에 국한되어 있으며 MoE(Mixture of Experts) 구조에서의 scaling behavior는 다를 수 있다. MoE는 파라미터 수는 많지만 활성화 되는 파라미터 수는 적기 때문에 가중치와 활성화 오차의 비율이 본 논문의 결과와 다르게 나타날 가능성이 높다. 또한, 4-bit 이하의 초저비트(Ternary 등) 환경에서도 이러한 통합 Scaling Law가 유효할지는 추가 연구가 필요하다.

결론적으로, 기존의 QAT 연구들이 활성화 함수의 Outlier 억제에만 집중했다면, 본 연구는 데이터 양이 많아질수록 가중치 양자화 오차의 영향력이 커진다는 점을 밝힘으로써, 향후 QAT 알고리즘 설계 시 가중치와 활성화를 동시에 최적화하는 균형 잡힌 접근법이 필요함을 제시하였다.

## 📌 TL;DR

본 논문은 LLM의 4-bit QAT 성능을 예측하기 위해 모델 크기($N$), 훈련 토큰 수($D$), 양자화 세밀도($G$)를 통합한 새로운 Scaling Law를 제안하였다. 실험을 통해 $\delta_p \propto D^{\gamma_D} G^{\gamma_G} N^{-\gamma_N}$ 의 관계를 입증하였으며, FC2Proj 레이어의 Outlier가 성능 저하의 핵심 원인임을 밝혀내어 8-bit mixed-precision 적용 시 성능을 크게 개선할 수 있음을 보여주었다. 이 연구는 효율적인 저비트 모델 설계를 위한 이론적 가이드라인을 제공하며, 향후 가중치와 활성화 오차를 동시에 최적화하는 QAT 연구의 중요성을 강조한다.