# Revisiting Bayesian Model Averaging in the Era of Foundation Models

Mijung Park (2025)

## 🧩 Problem to Solve

최근 Foundation Model(기반 모델)의 발전으로 이미지 및 텍스트 분류 성능이 비약적으로 향상되었으나, 각 타겟 데이터셋에 맞춰 모델 전체를 Fine-tuning하는 것은 막대한 계산 비용과 높은 $\text{CO}_2$ 배출량을 초래한다. 이에 대한 대안으로 여러 모델을 앙상블(Ensemble)하는 방법이 제안되었으나, 기존의 앙상블 방식들은 단순히 출력값을 동일한 가중치로 평균 내거나 모델 파라미터를 평균 내는 방식과 같은 휴리스틱(Heuristic)한 접근에 의존하는 경향이 있다.

본 논문의 목표는 고전적인 Bayesian Model Averaging(BMA) 패러다임을 재해석하여, 사전 학습된 Foundation Model이나 가볍게 Fine-tuning된 모델들을 원칙적으로 앙상블함으로써 분류 성능을 높이는 것이다. 특히, 거대 모델의 특성상 계산 불가능한(Intractable) 부분들을 해결하여 실용적인 BMA 및 최적화 가능한 모델 평균화(Optimizable Model Averaging, OMA) 기법을 제안하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Foundation Model의 파라미터는 고정(Frozen)하고, 그 뒤에 붙는 가벼운 선형 분류기(Linear Classifier)만을 학습 가능하게 하여 BMA를 적용 가능한 형태로 만드는 것이다.

1. **BMA의 실용적 구현**: 거대 모델의 전체 파라미터가 아닌, 선형 분류기의 가중치에 대해서만 Posterior(사후 확률)를 계산하여 각 모델의 기여도를 결정하는 원칙적인 앙상블 방법을 제시한다.
2. **OMA (Optimizable Model Averaging) 제안**: 학습 데이터와 검증 데이터 간의 분포 차이(Distribution Shift)가 크거나 레이블이 없는 경우를 위해, 예측의 '놀람(Surprise)', 즉 기대 엔트로피(Expected Entropy)를 최소화하는 방향으로 앙상블 가중치를 직접 최적화하는 OMA 기법을 제안한다.
3. **계산 효율성**: 전체 모델을 Fine-tuning하고 가중치를 평균 내는 기존 방식(예: Model Soups)에 비해, Frozen Feature를 사용함으로써 계산 비용을 획기적으로 줄이면서도 경쟁력 있는 성능을 달성한다.

## 📎 Related Works

기존의 모델 병합(Model Merging) 접근 방식은 크게 두 가지로 나뉜다. 첫 번째는 동일한 초기값에서 출발하여 독립적으로 최적화된 모델들의 가중치를 평균 내는 방식(예: Model Soups)이다. 두 번째는 여러 모델의 출력값을 앙상블하는 방식이다.

본 연구의 BMA 및 OMA는 출력값 앙상블의 일종으로 볼 수 있으나, 다음과 같은 차별점을 가진다.
- **비휴리스틱 가중치**: 단순 평균이 아니라 모델의 사후 확률이나 엔트로피 최적화를 통해 가중치를 결정한다.
- **낮은 비용**: 모델 전체를 Fine-tuning할 필요 없이, Frozen Feature 기반의 선형 헤드만 학습시키거나 가중치만 최적화하므로 훨씬 경제적이다.
- **아키텍처 유연성**: 가중치 평균 방식과 달리 서로 다른 아키텍처를 가진 모델들 간의 앙상블이 가능하다.

## 🛠️ Methodology

### 1. Bayesian Model Averaging (BMA)
BMA는 새로운 데이터 $x^*$에 대해 여러 후보 모델 $\{M_l\}_{l=1}^L$의 사후 확률을 가중치로 사용하여 예측한다.

$$p(y^*|x^*, D) = \sum_{l=1}^{L} p(y^*|x^*, M_l, D)p(M_l|D)$$

#### 가중치 계산을 위한 근사 (Laplace Approximation)
거대 모델에서 Marginal Likelihood(주변 가능도) $p(D|M_l)$를 계산하는 것은 분석적으로 불가능하다. 이를 해결하기 위해 본 논문은 다음과 같은 전략을 취한다.
- **Frozen Feature**: Foundation Model $\phi_l$은 고정하고, 선형 분류기 $w_l$만 학습 변수로 취급한다.
- **Laplace Approximation**: 사후 분포를 MAP(Maximum A Posteriori) 추정치 $w_{map}$을 중심으로 하는 다변량 가우시안 분포로 근사한다.
- **Block Diagonal Hessian**: 전체 Hessian 행렬의 크기가 너무 커서($10^6 \times 10^6$) 메모리에 올릴 수 없으므로, 대각 블록이 지배적이라는 가정을 통해 Block Diagonal Hessian 접근법을 사용한다.

최종적으로 근사된 로그 주변 가능도는 다음과 같이 계산된다.
$$\log p(D|M_l) \approx \log p(D|w_{map,l}, M_l) - \frac{1}{2} w_{map,l}^T S_\alpha^{-1} w_{map,l} - \frac{1}{2} \log |HS_\alpha + I|$$

### 2. Optimizable Model Averaging (OMA)
데이터 분포 변화가 심해 BMA의 사후 확률이 신뢰하기 어렵거나, 레이블이 부족한 경우를 위해 OMA를 제안한다. OMA의 핵심은 앙상블된 모델의 예측 결과에서 발생하는 '기대 엔트로피'를 최소화하는 가중치 $\beta_l$을 찾는 것이다.

#### 목적 함수
검증 데이터셋 $D_v$에 대해 다음의 손실 함수를 최소화한다.
$$L(\{\beta_l\}) = -\frac{1}{M} \sum_{m=1}^M \sum_{c=1}^C E \left[ \left( \sum_{l=1}^L \beta_l \cdot p(y^*_m=c|x^*_m, M_l) \right) \log \left( \sum_{l=1}^L \beta_l \cdot p(y^*_m=c|x^*_m, M_l) \right) \right]$$

여기에 사전 믿음 $\beta_{0,l}$에 대한 정규화 항 $\lambda \sum (\beta_l - \beta_{0,l})^2$을 추가하여 최적화하며, 가중치의 합이 1이 되고 $\beta_l \ge 0$이 되도록 제약 조건을 부여한다.

## 📊 Results

### 실험 설정
- **이미지 분류**: OpenCLIP 모델 8종을 특성 추출기로 사용. ImageNet-1K 및 OOD 데이터셋(Img-V2, Img-R, Img-sketch, Img-A, ObjNet)에서 평가.
- **텍스트 분류**: BERT-base, BERT-large, Funnel-transformer 모델 사용. GLUE 벤치마크(MRPC, RTE, CoLA, SST-2)에서 평가.
- **비교 대상**: Simple Output Averaging, Model Soups, CoCa, EVA02-L.

### 주요 결과
1. **이미지 분류**:
   - BMA는 ImageNet-1K(In-distribution)에서 단순 평균보다 훨씬 높은 성능을 보였다.
   - 분포 변화가 심한 OOD 데이터셋(Img-R, Img-sketch 등)에서는 OMA가 단순 평균보다 최대 2.4% 성능 향상을 보였으며, 일부 지표에서는 전체 Fine-tuning 기반의 Model Soups보다 우수한 성능을 기록했다.
   - Zero-shot 모델과 MAP 추정 모델을 OMA로 결합했을 때 가장 좋은 성능이 나타났다.

2. **텍스트 분류**:
   - OMA는 모든 GLUE 데이터셋에서 단순 출력 평균 및 단일 최적 모델보다 높은 성능을 보였다. (예: SST-2에서 96.62% 달성)
   - 학습된 가중치 $\beta$는 개별 모델의 성능과 강한 상관관계를 보였다.

3. **계산 비용**:
   - Model Soups나 CoCa 같은 방식은 수천 개의 TPU 칩을 사용하여 며칠간 학습해야 하지만, 제안 방법은 단일 GPU(RTX 4090)에서 수 시간 내에 완료 가능하다.

## 🧠 Insights & Discussion

본 논문은 Foundation Model 시대에 BMA를 어떻게 실용적으로 적용할 수 있는지를 보여주었다.

- **BMA vs OMA**: BMA는 모델의 복잡도(Complexity)에 패널티를 부여하여 단순하면서도 가능도가 높은 모델에 높은 가중치를 주는 반면, OMA는 복잡도와 무관하게 예측의 불확실성(Entropy)을 줄이는 방향으로 가중치를 설정한다. 따라서 OMA는 데이터 분포가 크게 변한 상황에서 더 유연하게 대처할 수 있다.
- **한계점**: BMA 프레임워크는 특성 추출기의 유용성이 검증된 이미지 도메인 외에는 적용이 어려울 수 있으며, OMA는 BMA와 달리 인식론적 불확실성(Epistemic Uncertainty)을 명시적으로 고려하지 않는다는 한계가 있다.
- **비판적 해석**: 본 연구는 거대 모델 전체를 튜닝하는 대신 선형 헤드만 조정하는 방식이 매우 효율적임을 입증했다. 이는 향후 더 거대한 모델들이 등장하더라도, 전체를 재학습시키지 않고 앙상블 가중치 최적화만으로 성능을 끌어올릴 수 있는 지속 가능한 경로를 제시한다.

## 📌 TL;DR

본 논문은 거대 Foundation Model들을 효율적으로 앙상블하기 위해, **선형 분류기 수준에서 사후 확률을 계산하는 BMA**와 **예측 엔트로피를 최소화하는 OMA** 기법을 제안한다. 실험 결과, 이 방법들은 매우 적은 계산 비용으로도 단순 평균보다 높은 성능을 내며, 특히 분포 외(OOD) 데이터에 대해 강력한 성능을 보였다. 이는 에너지 집약적인 전체 Fine-tuning 없이도 모델들의 시너지를 낼 수 있는 실용적인 방법론이다.