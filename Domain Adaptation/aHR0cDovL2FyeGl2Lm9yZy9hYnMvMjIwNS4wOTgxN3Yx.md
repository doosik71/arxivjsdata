# MiDAS: Multi-integrated Domain Adaptive Supervision for Fake News Detection

Abhijit Suprem, Calton Pu (2022)

## 🧩 Problem to Solve

본 논문은 코로나19(COVID-19) 팬데믹 상황에서 급증한 가짜 뉴스, 즉 '인포데믹(infodemic)' 현상을 해결하기 위한 가짜 뉴스 탐지 방법을 다룬다. 가짜 뉴스의 핵심적인 문제는 시간이 흐름에 따라 데이터의 분포가 변하는 'Concept Drift' 현상이 발생한다는 점이다. 이로 인해 특정 시점이나 특정 도메인의 데이터로 학습된 기존의 고성능 모델들이 새로운 도메인의 데이터나 최신 가짜 뉴스 샘플에 적용되었을 때 일반화 성능이 급격히 떨어지는 문제가 발생한다.

따라서 본 연구의 목표는 여러 도메인에서 학습된 기존의 가짜 뉴스 탐지 모델들이 있을 때, 새로운 입력 샘플에 대해 가장 적합한(best-fit) 모델을 동적으로 선택하여 분류 정확도를 높이는 적응형 결정 모듈(adaptive decision module)을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Local Lipschitz Smoothness**를 사용하여 새로운 샘플과 각 모델의 학습 데이터 간의 관련성(relevancy)을 측정하는 것이다. 모델이 특정 샘플 주변의 임베딩 공간에서 부드러운(smooth) 특성을 보인다면, 해당 모델이 그 영역의 데이터를 충분히 학습했음을 의미하며, 결과적으로 해당 샘플에 대해 더 높은 예측 정확도를 가질 것이라는 직관에 기반한다.

이를 구현하기 위해 MiDAS는 다음 두 가지 핵심 구성 요소를 제안한다. 첫째, 서로 다른 도메인의 데이터를 통합하여 비교 가능하게 만드는 **Domain-invariant Encoder**를 구축한다. 둘째, 이 불변 표현 공간 위에서 각 모델의 국소적 Lipschitz 상수를 계산하여 최적의 모델을 선택하는 **Adaptive Model Selector**를 구현한다.

## 📎 Related Works

기존의 도메인 적응(Domain Adaptation) 연구들은 주로 소스 도메인에서 타겟 도메인으로 맵핑하거나, 적대적 학습을 통해 도메인 불변 표현을 학습하여 단일 분류기를 만드는 방식에 집중하였다. 또한, Snorkel이나 EEWS와 같은 약지도 학습(Weak Supervision) 방식은 여러 레이블링 함수(labeling functions)를 조합하여 최적의 레이블을 추정하려 했다.

그러나 이러한 방식들은 각 도메인 모델의 가중치를 정적으로 부여하거나, 단순히 거리 기반의 가중치를 사용하는 한계가 있다. MiDAS는 모델의 예측 결과뿐만 아니라 임베딩 공간의 기하학적 특성인 '부드러움(smoothness)'을 직접 측정함으로써, 새로운 샘플에 대해 어떤 모델이 가장 신뢰할 수 있는지를 동적으로 결정한다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
MiDAS의 전체 구조는 단일 인코더와 다중 디코더, 그리고 기존에 미세 조정된(fine-tuned) 소스 모델(SM)들로 구성된다. 과정은 다음과 같다.
1. **Domain Invariant Encoder ($E$)**: 모든 소스 도메인의 데이터를 입력받아 도메인에 상관없는 공통의 불변 표현(invariant representation)을 생성한다.
2. **Decoders ($D_k$)**: 불변 표현을 다시 각 소스 모델 $f_k$가 이해할 수 있는 도메인 특화 표현으로 복원한다. 이는 기존 모델들을 그대로 사용하기 위해 SentencePiece 토큰 형태로 복원하는 과정을 포함한다.
3. **Adaptive Selection**: 불변 표현 공간에서 샘플 주변의 국소적 Lipschitz smoothness를 측정하여 가장 낮은 $L$ 값을 가진 모델을 선택한다.

### 훈련 목표 및 손실 함수
인코더 $E$를 학습시키기 위해 적대적 판별자(Discriminator, $D'$)와 Gradient Reversal Layer(GRL)를 사용한다. 판별자는 불변 표현이 어느 도메인에서 왔는지 맞추려 하고, 인코더는 판별자를 속이도록 학습된다.

$$ \text{Loss} = \min_{E,R(D')} -\sum_{i=1}^k \mathbb{E}_{(x,y)\sim X_k} [l(D'(R(E(x))), k)] $$

여기서 $R$은 GRL이며, 순전파 때는 항등 함수로 동작하고 역전파 때는 그래디언트에 $\lambda = -1$을 곱해 전달한다. 또한 디코더는 BERT/AlBERT의 Masked Language Modeling(MLM) 손실 함수를 사용하여 원문 토큰을 복원하도록 학습된다.

### Randomized Lipschitz Smoothness 및 모델 선택
MiDAS는 모델 $f_k$가 Lipschitz 연속적이라는 정의에서 출발한다. 즉, 두 입력의 차이에 비해 출력의 차이가 일정 상수 $L$에 의해 유계(bounded)될 때 이를 Lipschitz smooth하다고 한다.

$$ |\text{Pr}(f_k(x_1) = C) - \text{Pr}(f_k(x_2) = C)| \le L_k \cdot \theta(x_1, x_2) $$

새로운 샘플 $x'$에 대해, MiDAS는 $x'$ 주변의 $\epsilon$-Ball 내에서 $N$개의 무작위 점 $x_r$을 샘플링하여 다음 식을 통해 가장 부드러운 모델을 찾는다.

$$ \arg \min_k \max_{\theta(x', x_r) \le \epsilon} \left| \frac{1}{N} \sum_{r=1}^N \frac{\theta(\text{Pr}(f_k(x')), \text{Pr}(f_k(x_r)))}{\theta(x', x_r)} \right| $$

여기서 $\epsilon$은 각 모델의 클래터 중심(cluster center)에서 $m$개의 최근접 이웃을 통해 계산된 국소 $L$ 값의 최댓값의 역수($\epsilon = 1/\max(L_k)$)로 결정된다. 계산된 $L'_k$가 $1/\epsilon$보다 크면 해당 모델은 예측을 기권(abstain)하고, 작은 모델들 중 최적의 모델을 선택하여 최종 레이블을 결정한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 9개의 서로 다른 코로나19 가짜 뉴스 데이터셋을 사용하였다.
- **평가 방법**: Leave-one-out 방식을 채택하여, 8개 데이터셋으로 모델과 인코더를 학습시키고 나머지 1개의 보지 못한(drifted) 데이터셋에 대해 테스트하였다.
- **비교 대상**: Equal-weighted Ensemble, Snorkel, EEWS, KMP-model, 그리고 해당 데이터셋으로 직접 학습한 Oracle 모델을 기준으로 비교하였다.

### 주요 결과
- **정량적 성과**: MiDAS는 거의 모든 데이터셋에서 다른 베이스라인 모델들을 압도하였다. 특히 단순 앙상블(Ensemble) 대비 평균 30% 이상의 정확도 향상을 보였다.
- **일반화 능력**: Concept Drift가 심한 상황에서도 MiDAS는 각 샘플에 최적화된 모델을 선택함으로써 Oracle 모델의 성능에 근접하는 결과를 냈다.
- **Ablation Study**: MLM 마스킹 적용, Center Loss 추가, 판별자 손실 가중치 상향 조정 등이 순차적으로 성능 향상에 기여함을 확인하였다. 특히 MLM 적용이 엔드-투-엔드 정확도 향상에 유의미한 영향을 주었다.

## 🧠 Insights & Discussion

### 강점 및 분석
MiDAS의 가장 큰 강점은 기존에 이미 학습된 모델들을 수정하지 않고 그대로 사용할 수 있는 'Plug-and-play' 구조라는 점이다. 도메인 불변 인코더를 통해 서로 다른 도메인의 샘플들을 동일한 공간에 투영함으로써, 모델 간의 '부드러움'을 동일한 기준에서 비교할 수 있게 한 점이 주효했다. t-SNE 시각화 결과, MiDAS 적용 후 도메인 간 레이블 중첩(label overlap)이 줄어들고 클래스별 클러스터링이 명확해짐이 확인되었다.

### 한계 및 비판적 해석
- **하이퍼파라미터 $m$의 의존성**: 최근접 이웃의 수 $m$에 따라 $\epsilon$-Ball의 반지름이 결정되며, 이는 정확도(Accuracy)와 커버리지(Coverage) 사이의 트레이드오프를 발생시킨다. $m$이 너무 작으면 정확도는 높으나 예측 가능한 샘플 수가 적고, $m$이 너무 크면 커버리지는 넓어지나 정확도가 떨어진다.
- **구조적 제약**: 현재 구조는 기존 모델의 입력 형식을 맞추기 위해 디코더를 통해 토큰을 복원해야 한다. 저자들도 언급했듯이, 인코더와 소스 모델을 함께 학습시킨다면 복원 과정 없이 직접 임베딩을 사용함으로써 더 높은 성능과 효율성을 얻을 수 있을 것이다.

## 📌 TL;DR

MiDAS는 데이터 분포가 계속 변하는 가짜 뉴스 탐지 환경에서, **도메인 불변 표현(Domain-invariant representation)**과 **국소적 Lipschitz 부드러움(Local Lipschitz Smoothness)**을 이용해 새로운 샘플에 가장 적합한 모델을 동적으로 선택하는 프레임워크이다. 9개의 데이터셋 실험을 통해 기존 앙상블 및 약지도 학습 방법론 대비 월등한 일반화 성능을 입증하였으며, 이는 향후 실시간 가짜 뉴스 필터링 시스템 및 모델 일반화 연구에 중요한 기여를 할 것으로 보인다.