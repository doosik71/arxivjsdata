# Few-shot Metric Learning: Online Adaptation of Embedding for Retrieval

Deunsol Jung, Dahyun Kang, Suha Kwak, and Minsu Cho (2022)

## 🧩 Problem to Solve

본 논문은 딥 메트릭 러닝(Deep Metric Learning)에서 학습된 임베딩 함수가 소스 도메인과 타겟 도메인 사이에 상당한 도메인 간극(Domain Gap)이 존재할 때, 학습되지 않은 새로운 클래스(Unseen Classes)로 일반화되지 못하는 문제를 해결하고자 한다.

일반적으로 메트릭 러닝은 유사한 객체를 임베딩 공간 내의 가까운 점으로 매핑하는 임베딩 함수를 학습하는 것을 목표로 한다. 하지만 기존 방식들은 학습 단계에서 보지 못한 클래스에 대해 일반화 성능이 떨어지는 경향이 있으며, 특히 타겟 도메인의 특성이 소스 도메인과 크게 다를 경우 성능 저하가 심각하다.

따라서 본 논문은 타겟 도메인의 매우 적은 수의 레이블된 데이터(Few-shot)만을 사용하여 임베딩 함수를 온라인으로 적응(Adaptation)시키는 **Few-shot Metric Learning (FSML)**이라는 새로운 문제 정의와 해결 방법을 제안한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Few-shot Metric Learning 문제 정의**: 매우 적은 양의 타겟 클래스 데이터를 사용하여 임베딩 함수를 온라인으로 적응시키는 새로운 문제 프레임워크를 제안하였다.
2. **Channel-Rectifier Meta-Learning (CRML) 제안**: 중간 레이어의 채널을 조정하여 메트릭 공간을 효과적으로 적응시키는 메타 러닝 방법을 제안하였다. 이는 모든 파라미터를 업데이트할 때 발생하는 과적합(Overfitting) 문제를 방지하면서도, 출력층만 튜닝할 때의 성능 부족 문제를 동시에 해결한다.
3. **새로운 데이터셋 miniDeepFashion 제안**: 속성(Attribute)에 따라 유사도 관계가 변하는 다중 속성 이미지 검색 데이터셋을 구축하여, 온라인 적응의 필요성을 입증하였다.
4. **광범위한 실험적 검증**: 표준 벤치마크 및 크로스 도메인 실험을 통해 제안 방법이 기존 메트릭 러닝 및 퓨샷 분류(Few-shot Classification) 방식보다 검색 성능이 우수함을 증명하였다.

## 📎 Related Works

### Metric Learning

기존의 메트릭 러닝은 Contrastive loss, Triplet loss와 같은 쌍 기반(Pair-wise) 손실 함수나 Proxy 기반 손실 함수를 사용하여 일반화 성능을 높이는 데 집중하였다. 이를 논문에서는 Conventional Metric Learning(DML)이라 칭하며, 이들은 학습 시 보지 못한 클래스에 대해 일반화하는 것을 목표로 하지만, 도메인 간극이 클 경우 한계가 명확하다.

### Few-shot Classification

퓨샷 분류는 적은 데이터로 클래스를 구분하는 결정 경계(Decision Boundary)를 만드는 데 집중한다. 본 논문은 퓨샷 분류와 퓨샷 메트릭 러닝이 "적은 데이터로 적응한다"는 점은 같으나, 목적 함수와 평가 프로토콜이 완전히 다름을 강조한다. 퓨샷 분류는 단순히 클래스를 맞추는 것에 집중하므로, 검색(Retrieval) 작업에 필수적인 객체 간의 상대적 거리 관계를 보존하는 능력이 부족하다.

## 🛠️ Methodology

### 1. Problem Formulation

본 논문은 $N$-way $K$-shot 설정을 따른다.

- **Support Set ($S$)**: 타겟 클래스 $N$개에 대해 각각 $K$개의 레이블된 샘플을 포함한다.
- **Prediction Set ($P$)**: 동일한 타겟 클래스에서 추출된 $M$개의 보지 못한 샘플을 포함한다.
목표는 $S$를 이용하여 임베딩 모델을 적응시켜, $P$ 내의 인스턴스 검색 성능을 극대화하는 것이다. 이를 위해 메타 학습(Meta-learning)의 에피소드 훈련(Episodic Training) 방식을 채택한다.

### 2. Baselines

논문은 비교를 위해 세 가지 베이스라인을 구축하였다.

- **Simple Fine-Tuning (SFT)**: 사전 학습된 모델을 타겟 서포트 셋 $S$에 대해 단순하게 파인튜닝한다.
- **Model-Agnostic Meta-Learning (MAML)**: 퓨샷 적응에 최적화된 초기 파라미터 $\theta_0$를 학습한다. Inner loop에서 $S$를 통해 업데이트하고, Outer loop에서 $P$의 손실을 통해 초기값을 업데이트한다.
- **Meta-Transfer Learning (MTL)**: 대부분의 레이어를 동결하고, 마지막 FC 레이어의 초기값과 각 컨볼루션 레이어의 채널 스케일링/시프팅 파라미터 $\Phi = \{(\gamma_l, \beta_l)\}$를 학습한다.

### 3. Channel-Rectifier Meta-Learning (CRML)

CRML은 SFT와 MAML의 과적합 문제와 MTL의 낮은 표현력 문제를 해결하기 위해, 중간 레이어의 채널을 조정하는 **Channel Rectifier**를 온라인으로 적응시키는 방식이다.

**핵심 구조:**
사전 학습된 임베딩 모델 $f$의 모든 파라미터는 동결시킨다. 대신, 각 컨볼루션 레이어에 다음과 같은 채널 수정 모듈을 적용한다.
$$\text{Conv}_l^{CSS}(X; W_l, b_l, \gamma_l, \beta_l) = (W_l \odot \gamma_l) * X + (b_l + \beta_l)$$
여기서 $\gamma_l$은 채널별 곱셈(scaling)을, $\beta_l$은 채널별 덧셈(shifting)을 수행한다.

**학습 및 추론 절차:**

1. **Meta-Training**: 채널 수정 파라미터 $\Phi = \{(\gamma_l, \beta_l)\}$의 최적 초기값 $\Phi_0$를 메타 학습한다.
   - **Inner loop**: 서포트 셋 $S^{mtr}$을 사용하여 $\Phi_0$를 $\Phi_1$으로 업데이트한다.
   - **Outer loop**: 예측 셋 $P^{mtr}$에서의 손실을 바탕으로 $\Phi_0$를 업데이트한다.
2. **Meta-Testing**: 새로운 타겟 클래스가 주어지면, 학습된 $\Phi_0$에서 시작하여 서포트 셋 $S$를 이용해 $\Phi$를 빠르게 파인튜닝한 후 검색을 수행한다.

## 📊 Results

### 실험 설정

- **데이터셋**: miniImageNet, CUB-200-2011, MPII (연속적 레이블), miniDeepFashion (다중 속성).
- **지표**: mAP, Recall@k (이산 레이블), mPD, nDCG (연속 레이블).
- **모델**: ResNet-18 기반 임베딩 모델 (임베딩 크기 128), Multi-similarity loss 사용.

### 주요 결과

1. **FSML의 효과**: 모든 퓨샷 메트릭 러닝 방법이 기존의 DML보다 우수한 성능을 보였다. 특히 단 5개의 샘플(5-shot)만으로도 검색 품질을 크게 높일 수 있음을 확인하였다.
2. **CRML의 우수성**: CRML은 miniImageNet 및 크로스 도메인 설정에서 MAML과 SFT를 큰 차이로 앞섰다. 이는 전체 파라미터를 업데이트하는 방식보다 채널을 수정하는 방식이 퓨샷 상황에서 과적합을 훨씬 잘 방지하기 때문이다.
3. **도메인 간극과의 상관관계**: 도메인 간극이 클수록(예: miniImageNet $\rightarrow$ CUB 크로스 도메인) 온라인 적응을 통한 성능 향상 폭(Adaptation Growth Rate)이 더 크게 나타났다.
4. **분류 vs 검색**: 퓨샷 분류 모델은 분류 정확도는 높았으나, 검색 성능(mAP, Recall@1)은 FSML 방법론들에 비해 현저히 낮았다. 이는 분류를 위한 결정 경계 학습과 검색을 위한 메트릭 공간 학습이 서로 다른 문제임을 시사한다.
5. **miniDeepFashion**: 속성에 따라 유사도가 변하는 이 데이터셋에서 DML은 단순히 색상이나 모양에 의존하는 경향을 보였으나, CRML은 온라인 적응을 통해 타겟 속성(예: 카테고리)에 맞는 정확한 검색 결과를 도출하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 메트릭 러닝에서 간과되었던 '온라인 적응'의 중요성을 체계적으로 분석하였다. 특히, 모델 전체를 튜닝하는 것은 과적합을 유발하고, 출력층만 튜닝하는 것은 표현력 부족을 야기한다는 점을 지적하며, 그 절충안으로 **중간 레이어의 채널 수정(Channel Rectification)**이라는 효율적인 경로를 제시하였다.

### 한계 및 비판적 해석

- **계산 복잡도**: 메타 학습 과정(특히 MAML 계열)은 훈련 시간이 오래 걸리며, 추론 시에도 서포트 셋에 대한 그래디언트 업데이트 단계가 필요하므로 실시간 응답성이 중요한 시스템에서는 오버헤드가 될 수 있다.
- **데이터셋 의존성**: miniDeepFashion에서 성능 향상이 있었으나, 여전히 DML 대비 절대적인 수치는 낮다. 이는 속성 간의 '시맨틱 스위치(Semantic Switch)'가 일어나는 매우 어려운 문제임을 보여주며, 단순한 채널 수정 이상의 더 강력한 적응 기법이 필요할 수 있음을 암시한다.

## 📌 TL;DR

본 논문은 소수 데이터만으로 임베딩 공간을 타겟 도메인에 맞게 최적화하는 **Few-shot Metric Learning (FSML)** 문제를 제안하고, 이를 해결하기 위해 중간 레이어의 채널을 조정하는 **Channel-Rectifier Meta-Learning (CRML)** 방법을 제시하였다. 실험 결과, CRML은 도메인 간극이 큰 상황에서 특히 강력한 성능을 보였으며, 이는 기존의 퓨샷 분류 방식으로는 달성할 수 없는 검색 성능의 향상을 가져왔다. 이 연구는 향후 도메인 적응형 이미지 검색 및 정밀한 객체 매칭 시스템 구축에 중요한 기초가 될 것으로 보인다.
