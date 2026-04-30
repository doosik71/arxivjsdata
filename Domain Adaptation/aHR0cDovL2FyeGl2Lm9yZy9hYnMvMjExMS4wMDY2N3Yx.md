# Unsupervised Domain Adaptation with Adapter

Rongsheng Zhang, Yinhe Zheng, Xiaoxi Mao, Minlie Huang (2021)

## 🧩 Problem to Solve

본 논문은 사전 학습된 언어 모델(Pre-trained Language Model, 이하 PrLM)을 이용한 비지도 도메인 적응(Unsupervised Domain Adaptation, 이하 UDA) 과정에서 발생하는 두 가지 핵심 문제를 해결하고자 한다.

첫째, 소규모의 도메인 특화 코퍼스로 PrLM의 모든 파라미터를 미세 조정(Fine-tuning)할 경우, 모델이 특정 도메인으로 과도하게 편향되어 PrLM이 원래 보유하고 있던 일반적인 지식(Generic Knowledge)이 왜곡되는 현상이 발생한다. 이는 도메인 간 전이 가능한 특징(Transferable Features)을 포착하는 능력을 저하시켜 결국 UDA 성능의 하락으로 이어진다.

둘째, 각 도메인마다 전체 모델 파라미터를 미세 조정하여 개별 모델을 구축하고 배포하는 것은 계산 비용 및 저장 공간 측면에서 매우 비효율적이다.

따라서 본 연구의 목표는 PrLM의 일반적 지식을 보존하면서도 효율적으로 타겟 도메인에 적응할 수 있는 Adapter 기반의 UDA 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 PrLM의 기존 파라미터를 고정(Freeze)한 상태에서, 학습 가능한 소규모의 Adapter 모듈만을 추가하여 도메인 적응을 수행하는 것이다. 이를 통해 모델의 파라미터 효율성을 높이는 동시에 일반적 지식의 왜곡을 방지한다.

특히, 단순한 Adapter 적용을 넘어 **2단계 적응 과정(Two-step Adaption)**을 도입하였다. 첫 단계에서는 소스 도메인과 타겟 도메인의 데이터를 모두 사용하여 Adapter가 도메인 간 공통적이고 전이 가능한 특징을 학습하도록 하고, 두 번째 단계에서는 소스 도메인의 레이블된 데이터를 통해 특정 태스크에 최적화하도록 설계하였다.

## 📎 Related Works

UDA 연구는 크게 두 가지 방향으로 분류된다. 하나는 특성 공간을 확장하거나 새로운 손실 함수를 설계하는 모델 기반 방법(Model-based methods)이고, 다른 하나는 의사 레이블(Pseudo-labels)이나 데이터 선택 기법을 사용하는 데이터 기반 방법(Data-based methods)이다. 최근에는 PrLM을 활용하는 방식이 표준이 되었으나, 앞서 언급한 전체 파라미터 미세 조정으로 인한 지식 왜곡 문제가 한계점으로 지적된다.

Adapter 모듈은 주로 파라미터 효율적인 미세 조정을 위해 사용되어 왔으며, 최근에는 제로샷 교차 언어 전이(Zero-shot cross-lingual transfer) 작업 등에서 언어별 지식을 분리하는 용도로 활용되었다. 본 논문은 이러한 Adapter의 개념을 UDA에 적용하여, 언어 분리가 아닌 도메인 간 전이 가능한 공통 특징을 포착하는 방향으로 차별화를 두었다.

## 🛠️ Methodology

### 1. Adapter Architecture
본 논문에서는 Transformer 기반 PrLM의 각 레이어 내 Feed-Forward 서브레이어 뒤에 보틀넥(Bottleneck) 구조의 MLP 형태인 Adapter 모듈을 삽입한다.

Adapter의 구조는 다음과 같다:
1. **Down-project**: 입력 표현의 차원 $H$를 더 작은 차원 $m$으로 투영한다.
2. **Nonlinearity**: $GELU$ 활성화 함수를 적용한다.
3. **Up-project**: 다시 차원을 $H$로 복원한다.
4. **Residual Connection**: 입력값과 출력값을 더하는 잔차 연결을 적용하여 학습의 안정성을 높인다.

### 2. Two-step Adaption Process
PrLM의 기존 파라미터는 모든 과정에서 고정되며, 오직 Adapter와 헤드(Head) 부분만 학습한다.

**단계 1: 도메인 융합 학습 (Domain-fusion Training)**
소스 도메인 $D_s$와 모든 타겟 도메인 $D_{t_i}$가 섞인 통합 코퍼스를 사용하여 Adapter를 학습시킨다. 이때 학습 목표는 Masked Language Model (MLM) 손실 함수 $\mathcal{L}_{MLM}$을 최소화하는 것이다. 이 과정은 Adapter가 서로 다른 도메인 간의 전이 가능한 특징을 캡처하고 융합하도록 유도한다.

**단계 2: 태스크 미세 조정 (Task Fine-tuning)**
소스 도메인의 레이블된 데이터 $D_s$를 사용하여 태스크 전용 헤드(Task Head)와 Adapter를 함께 학습시킨다. 이때는 해당 태스크에 맞는 손실 함수 $\mathcal{L}_{Task}$ (예: 분류를 위한 Cross-Entropy Loss)를 사용한다. 이미 1단계에서 도메인 간 공통 특징을 학습했으므로, 소스 도메인에서 학습된 태스크 지식이 타겟 도메인으로 더 효과적으로 일반화될 수 있다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - SDA (Sentiment Domain Adaptation): 아마존 제품 리뷰 (Books, DVDs, Electronics, Kitchen)
    - XNLI (Cross-lingual NLI): 15개 언어의 NLI 데이터 (영어 $\to$ 타 언어)
- **모델**: RoBERTa-base (SDA용), XLM-R base (XNLI용)
- **비교 대상 (Baselines)**:
    - Full-FT: 전체 파라미터를 소스 도메인 데이터로 미세 조정
    - Full-TSA: 전체 파라미터를 사용하여 본 논문의 2단계 적응 과정을 수행
    - Ada-FT: Adapter를 사용하되, 도메인 융합 학습 없이 바로 태스크 미세 조정 수행
    - Ada-TSA: 제안 방법 (Adapter + 2단계 적응 과정)

### 주요 결과
- **정량적 성능**: 대부분의 설정에서 Ada-TSA가 가장 높은 정확도를 기록하였다.
    - SDA 데이터셋 평균 정확도: $\text{Full-FT}(92.10\%) \to \text{Ada-TSA}(93.31\%)$
    - XNLI 데이터셋 평균 정확도: $\text{Full-FT}(71.62\%) \to \text{Ada-TSA}(72.71\%)$
- **데이터 크기의 영향**: 소스 데이터($D_s$)가 매우 적을 때, Ada-FT는 Full-FT 대비 성능 향상이 미미했지만, 도메인 융합 단계가 포함된 Ada-TSA는 상당한 성능 향상을 보였다. 이는 소규모 데이터만으로는 무작위 초기화된 Adapter를 충분히 학습시키기 어렵지만, 도메인 융합 학습이 이를 보완함을 시사한다.
- **도메인 유사성의 영향**: 소스-타겟 도메인 간의 어휘 중첩도(Vocabulary Overlap)가 높을수록(SDA $\gg$ XNLI) 도메인 융합 학습의 효과가 더 크게 나타났다.

## 🧠 Insights & Discussion

본 논문은 Adapter 기반의 튜닝이 PrLM의 일반적 지식을 보존하면서도 도메인 특화 능력을 효율적으로 습득할 수 있음을 입증하였다. 특히 t-SNE 시각화 분석을 통해, 전체 모델을 튜닝하는 것보다 Adapter를 통해 도메인 융합 학습을 진행했을 때 은닉 표현(Hidden Representation)의 변화가 더 효과적으로 일어나며, 이는 타겟 도메인으로의 전이 성능 향상으로 이어진다는 것을 확인하였다.

다만, 본 연구는 도메인 간 유사성이 매우 낮은 경우(예: 완전히 다른 언어 간의 전이)에는 도메인 융합 학습의 이득이 상대적으로 적다는 한계를 보였다. 이는 Adapter가 캡처할 수 있는 '전이 가능한 특징'의 범위가 도메인 간의 기본적인 유사성에 의존하기 때문으로 해석된다.

## 📌 TL;DR

본 논문은 PrLM의 파라미터를 고정하고 학습 가능한 Adapter 모듈만을 사용하는 UDA 방법을 제안한다. 특히 **'도메인 융합 학습(MLM) $\to$ 태스크 미세 조정'**으로 이어지는 2단계 학습 전략을 통해, 지식 왜곡 없이 도메인 간 공통 특징을 효과적으로 포착하여 UDA 성능을 향상시켰다. 이 연구는 특히 데이터가 부족하거나 배포 효율성이 중요한 실제 산업 환경의 도메인 적응 문제에 중요한 해결책을 제시한다.