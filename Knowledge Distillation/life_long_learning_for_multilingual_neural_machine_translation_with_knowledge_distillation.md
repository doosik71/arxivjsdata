# Life-long Learning for Multilingual Neural Machine Translation with Knowledge Distillation

Yang Zhao, Junnan Zhu, Lu Xiang, Jiajun Zhang, Yu Zhou, Feifei Zhai, and Chengqing Zong (2022)

## 🧩 Problem to Solve

본 논문은 다국어 신경 기계 번역(Multilingual Neural Machine Translation, MNMT) 시스템에서 새로운 번역 작업이 순차적으로 추가될 때 발생하는 **Catastrophic Forgetting (CF, 치명적 망각)** 문제를 해결하고자 한다.

일반적인 MNMT 연구는 모든 언어 쌍의 학습 데이터를 동시에 사용할 수 있는 설정에서 진행되지만, 실제 환경에서는 다음과 같은 제약 조건이 존재한다.

- **순차적 작업 도착**: 새로운 번역 작업이 시간에 따라 하나씩 추가된다.
- **이전 데이터 접근 불가**: 데이터 프라이버시 및 보호 문제로 인해 이전에 학습했던 작업의 데이터를 다시 사용할 수 없다.

이러한 상황에서 새로운 작업으로 모델을 단순 미세 조정(Fine-tuning)하면 이전 작업의 성능이 급격히 저하되는 CF 현상이 발생하며, 모든 데이터를 저장하고 재학습하는 Joint training은 저장 공간 및 계산 비용 문제로 인해 불가능하다. 따라서 본 논문의 목표는 이전 데이터 없이도 기존 지식을 유지하면서 새로운 작업을 학습할 수 있는 **Life-long Learning (평생 학습)** 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Knowledge Distillation (지식 증류)**을 활용하여 이전 모델(Teacher)의 지식을 새로운 모델(Student)에게 전수함으로써 CF를 억제하는 것이다. 특히 MNMT의 두 가지 주요 시나리오에 따라 서로 다른 접근 방식을 제안한다.

1. **Incremental One-to-Many 시나리오**: 하나의 소스 언어를 여러 타겟 언어로 번역하는 경우, Teacher 모델의 다국어 출력값을 Student가 함께 학습하는 **Multilingual Distillation** 방법을 제안한다.
2. **Incremental Many-to-One 시나리오**: 여러 소스 언어를 하나의 타겟 언어로 번역하는 경우, 새로운 소스 언어의 토큰들이 기존 모델에서 $\text{UNK}$로 처리되어 지식 전수가 제대로 이루어지지 않는 **Extreme Partial Distillation** 문제를 정의하고, 이를 해결하기 위해 **Pseudo Input Distillation**과 **Reverse Teacher Distillation**이라는 두 가지 방법을 제안한다.

## 📎 Related Works

논문에서는 기존의 MNMT 접근 방식과 Life-long Learning 연구들을 소개한다.

- **MNMT**: 보편적인 인코더와 디코더를 사용하고 언어 지시자(Indicator)를 추가하는 방식(Johnson et al., 2017)이 널리 사용되지만, 이는 모든 데이터를 동시에 사용할 수 있다는 가정을 전제로 한다.
- **Life-long Learning**: 기존 연구들은 Replay-based, Regularization-based(예: EWC), Parameter isolation-based 방법으로 분류된다.
- **차별점**: 기존의 Life-long Learning 연구들은 주로 이미지 분류나 객체 탐지, 혹은 단순한 NLP 작업(감성 분석 등)에 집중되어 있었으며, MNMT와 같이 복잡한 시퀀스 생성 작업과 언어별 어휘(Vocabulary) 문제가 얽힌 incremental MNMT 시나리오는 충분히 다뤄지지 않았다.

## 🛠️ Methodology

### 1. Incremental One-to-Many Scenario

이 시나리오의 목표는 기존 모델 $\theta(X \Rightarrow Y_1, \dots, Y_n)$과 새 데이터 $D_{X \Rightarrow Y_{n+1}}$을 이용하여 $\theta(X \Rightarrow Y_1, \dots, Y_{n+1})$을 구축하는 것이다.

- **절차**:
    1. 학습된 모든 타겟 언어 $Y_i$에 대해 소스 문장 $X$에 지시자 $\langle X2Y_i \rangle$를 추가한다.
    2. Teacher 모델에 이를 입력하여 Beam Search를 통해 최적의 번역 결과 $Y_i$를 생성한다.
    3. Student 모델은 새로운 작업의 손실 함수와 Teacher가 생성한 이전 작업들의 결과를 함께 학습한다.

- **목표 함수**:
$$\mathcal{L}(\theta^{(X \Rightarrow Y_1, \dots, Y_{n+1})}) = \sum_{i=1}^{n} \log p(Y_i | X + \langle X2Y_i \rangle) + \log p(Y_{n+1} | X + \langle X2Y_{n+1} \rangle)$$

### 2. Incremental Many-to-One Scenario

이 시나리오에서는 새로운 소스 언어 $X_{n+1}$의 토큰이 기존 모델의 어휘집에 없어 $\text{UNK}$로 치환되는 **Extreme Partial Distillation** 문제가 발생한다. 이를 해결하기 위한 두 가지 방법은 다음과 같다.

#### A. Pseudo Input Distillation

새로운 소스 문장을 기존 모델이 이해할 수 있는 토큰으로 변환하여 입력하는 방식이다.

- **빈도 기반 매핑**: 각 언어의 어휘집 $V_{X_i}$를 빈도수 내림차순으로 정렬한다. 새로운 언어 $X_{n+1}$의 $j$번째 빈도 토큰을 기존 언어 $X_i$의 $j$번째 빈도 토큰으로 매핑하는 함수 $M(X_{n+1} \rightarrow X_i)$를 생성한다.
- **가상 입력 생성**: 새로운 문장 $X_{n+1}$의 토큰들을 이 매핑을 통해 기존 언어 $X_i$의 토큰들로 교체하여 가상 입력 $X^p_i$를 만든다.
- **학습**: $X^p_i$를 Teacher 모델에 넣어 얻은 결과 $Y_i$를 Student가 학습한다.

#### B. Reverse Teacher Distillation

역번역 모델을 Teacher로 활용하는 방식이다.

- **역모델 활용**: $\theta(Y \Rightarrow X_1, \dots, X_n)$ 형태의 역번역 모델을 Teacher로 사용한다.
- **절차**: 타겟 문장 $Y$에 지시자 $\langle Y2X_i \rangle$를 추가하여 입력하면, Teacher가 소스 언어 $X_i$에 해당하는 문장을 생성한다. Student는 이렇게 생성된 $(X_i, Y)$ 쌍을 통해 기존 지식을 학습한다.
- **업데이트**: 다음 작업을 위해 역번역 모델 또한 함께 업데이트(Reverse Student)한다.

## 📊 Results

### 실험 설정

- **데이터셋**: TED, CCMT-19, LDC, KFTT, WMT-17 등 총 12개의 번역 작업 사용.
- **모델**: Transformer-base 아키텍처, BPE 30K 적용.
- **평가 지표**: BLEU score.
- **비교 대상**: Single model, Joint Training (Upper Bound), Fine-tuning, EWC, Direct Distillation.

### 주요 결과

- **One-to-Many 시나리오**:
  - Fine-tuning은 CF 현상으로 인해 BLEU 점수가 급격히 하락(예: 28.51 $\rightarrow$ 1.08)한다.
  - **Multilingual Distillation**은 CF를 효과적으로 억제하며, Joint Training과 유사한 수준의 성능을 보였다. 특히 Beam search 방식이 Greedy search보다 우수한 결과를 냈다.
- **Many-to-One 시나리오**:
  - Direct Distillation은 $\text{UNK}$ 문제로 인해 Fine-tuning보다 오히려 성능이 더 떨어지는 결과가 나타났다.
  - **Reverse Teacher Distillation**이 가장 우수한 성능을 보였으며, Joint Training에 매우 근접한 수치를 기록했다.
  - Pseudo Input Distillation 역시 CF를 완화했지만, Reverse Teacher 방식보다는 성능이 낮았다.

## 🧠 Insights & Discussion

본 논문은 MNMT의 순차적 학습 환경에서 지식 증류가 CF를 방지하는 강력한 도구가 될 수 있음을 입증하였다.

**핵심 통찰**:
Many-to-One 시나리오에서 단순히 Teacher 모델의 출력을 복제하는 Direct Distillation이 실패하는 이유는 **입력 공간의 불일치** 때문이다. 새로운 언어의 토큰이 $\text{UNK}$로 처리되면 모델은 '의미'가 아닌 'UNK 처리 방식'만을 학습하게 된다. 이를 해결하기 위해 빈도 기반의 토큰 매핑(Pseudo Input)을 사용하거나, 입력-출력 방향을 완전히 뒤집어 타겟 언어를 기준으로 소스 문장을 생성(Reverse Teacher)하게 함으로써 실질적인 지식 전수를 가능케 했다.

**한계점**:

- **계산 비용**: Knowledge Distillation 과정에서 Teacher 모델의 추론 및 Student 모델의 추가 학습이 필요하므로 계산 오버헤드가 발생한다.
- **확장성**: 본 연구는 One-to-Many와 Many-to-One을 분리하여 다루었으며, 가장 복잡한 형태인 Many-to-Many 시나리오에 대해서는 다루지 않았다.

## 📌 TL;DR

본 논문은 이전 학습 데이터에 접근할 수 없는 환경에서 새로운 번역 언어를 추가할 때 발생하는 **치명적 망각(Catastrophic Forgetting)** 문제를 해결하기 위해 맞춤형 지식 증류(Knowledge Distillation) 기법을 제안한다. One-to-Many에서는 **Multilingual Distillation**을, Many-to-One에서는 $\text{UNK}$ 문제를 해결한 **Pseudo Input** 및 **Reverse Teacher Distillation**을 통해 Joint Training에 근접한 성능을 달성하였다. 이 연구는 데이터 프라이버시가 중요하거나 지속적으로 언어를 확장해야 하는 실제 MNMT 시스템 구축에 중요한 방법론을 제공한다.
