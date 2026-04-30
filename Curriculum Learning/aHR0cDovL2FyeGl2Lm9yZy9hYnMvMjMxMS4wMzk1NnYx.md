# Cup Curriculum: Curriculum Learning on Model Capacity

Luca Scharr, Vanessa Toborek (2023)

## 🧩 Problem to Solve

본 논문은 자연어 처리(NLP) 분야에서 모델의 용량(Model Capacity)을 조절하는 커리큘럼 학습(Curriculum Learning, CL)의 가능성을 탐구한다. 일반적으로 커리큘럼 학습은 학습 데이터의 순서나 태스크의 난이도를 조절하는 방식에 집중되어 있으며, 모델의 구조적 용량을 동적으로 변화시키는 연구는 NLP 분야에서 거의 이루어지지 않았다.

인간의 뇌 발달 과정에서 유아기와 청소년기에 시냅스 밀도가 크게 변화하는 현상이 학습 능력에 긍정적인 영향을 미친다는 점에 착안하여, 인공신경망에서도 모델 용량을 전략적으로 줄였다가 다시 늘리는 과정이 성능 향상을 가져올 수 있다는 가설을 세운다. 따라서 본 연구의 목표는 모델 용량을 '컵(Cup)' 모양의 곡선으로 변화시키는 **Cup Curriculum** 전략을 제안하고, 이것이 기존의 Early Stopping이나 단순한 Pruning 방식보다 우수한 성능을 보이는지 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델의 표현 능력(Expressiveness)을 먼저 감소시킨 후 다시 복구시키는 두 단계의 순차적 학습 과정을 통해 최적의 파라미터 설정을 찾는 것이다.

1. **Cup Curriculum 제안**: 모델 용량을 점진적으로 감소시키는 Pruning Phase와 다시 증가시키는 Growth Phase로 구성된 새로운 학습 프레임워크를 제안하였다.
2. **NLP 모델 용량 CL 분석**: NLP 분야, 특히 Transformer 아키텍처에서 모델 용량 기반의 커리큘럼 학습을 분석한 최초의 연구 중 하나이다.
3. **다양한 전략 실험 및 검증**: Rewinding, Initialization, Update scheme의 다양한 조합을 실험하여, 이 방식이 Early Stopping 대비 신뢰도 99% 수준에서 성능을 향상시키며 과적합(Overfitting)에 강건함을 입증하였다.

## 📎 Related Works

커리큘럼 학습(CL)은 벤지오(Bengio et al.)의 연구 이후 주로 학습 데이터의 구성에 집중되어 발전해 왔다. 모델 용량을 조절하는 방식의 CL은 일부 연구(Morerio et al., Sinha et al.)에서 Dropout 확률을 조절하거나 노이즈를 추가하는 방식으로 시도되었으나, 이는 NLP 분야의 연구가 아니었다. 

CNN 필터를 제거했다가 나중에 다시 도입하는 RePr(Prakash et al.) 방식이 제시된 바 있으나, 이는 필터 단위의 조절이었으며 본 논문과 같이 개별 가중치(Weight) 수준에서 Transformer 아키텍처에 적용한 사례는 드물다. 또한, 본 연구의 기반이 되는 Iterative Magnitude Pruning(IMP)은 가중치의 크기를 기준으로 중요도를 판단하여 제거하는 기법으로, Transformer 모델에서도 성공적으로 적용된 바 있다.

## 🛠️ Methodology

### 전체 파이프라인
Cup Curriculum은 크게 **Pruning Phase**와 **Growth Phase**라는 두 가지 단계로 나뉜다. 모델 용량의 변화 추이가 컵 모양(감소 후 증가)을 그리게 된다.

### 1. Pruning Phase (용량 감소 단계)
이 단계에서는 반복적인 Pruning 사이클을 통해 모델의 파라미터 수를 줄여나간다.
- **절차**: 학습 $\rightarrow$ Pruning $\rightarrow$ (선택적) Rewinding의 과정을 반복한다.
- **Pruning Criterion**: 가중치의 중요도를 판단하기 위해 'Magnitude Change'를 사용한다.
  $$c = \|w_c\| - \|w_i\|$$
  여기서 $w_c$는 현재 가중치 값이고, $w_i$는 초기 가중치 값이다.
- **Rewinding Schemes**: Pruning 후 남은 가중치들을 어떤 상태로 되돌릴지 결정한다.
    - `Initial`: 초기 상태 $\Theta_0$로 되돌림.
    - `Warm`: Warm-up 상태 $\Theta_{warm}$으로 되돌림.
    - `Best`: 해당 사이클에서 가장 성능이 좋았던 상태 $\Theta_{best, cycle}$로 되돌림.
    - `No`: 되돌리기 수행 안 함.

### 2. Growth Phase (용량 증가 단계)
Pruning 단계에서 제거되었던 가중치들을 다시 도입하여 모델 용량을 복구한다.
- **절차**: 가중치 도입 $\rightarrow$ 학습의 과정을 반복한다.
- **Initialization Schemes**: 재도입되는 가중치의 초기값을 어떻게 설정할지 결정한다.
    - `Original`: 초기 상태 $\Theta_0$ 값을 사용.
    - `Random`: 무작위 분포로 초기화.
    - `Old`: Pruning 직전의 값 $\Theta_{last, cycle}$을 사용.
    - `Top`: 해당 가중치가 제거되었던 사이클의 최적 상태 $\Theta_{best, cycle}$ 값을 사용.
- **Update Schemes**: 재도입된 가중치를 어떻게 업데이트할지 결정한다.
    - `Freezing`: 가장 최근에 도입된 가중치만 업데이트하고 나머지는 고정.
    - `Identical`: 모든 가중치에 동일한 학습률(LR)을 적용하는 표준 역전파.
    - `Dynamic`: 도입 시점에 따라 학습률을 다르게 적용.

## 📊 Results

### 실험 설정
- **데이터셋**: WikiText2
- **모델**: Transformer 아키텍처 (Small, Medium, Large 세 가지 크기)
- **지표**: Cross-entropy loss (Perplexity)
- **비교 대상**: Early Stopping, 표준 IMP

### 주요 결과
- **정량적 성능 향상**: 최적의 전략(Best Rewinding + Random Initialization + Identical Updating)을 사용했을 때, Early Stopping 대비 **0.5%에서 2%**의 성능 향상이 관찰되었다.
- **모델 크기와의 상관관계**: 모델의 크기가 커질수록 Cup Curriculum을 통한 성능 향상 폭이 더 커지는 경향을 보였다.
- **통계적 유의성**: Wilcoxon-Mann-Whitney 테스트 결과, 99%의 신뢰 수준($\alpha = 0.01$)에서 유의미한 개선이 확인되었다.
- **과적합 강건성**: Early Stopping은 과적합이 시작되는 지점에서 학습을 멈춰야 하지만, Cup Curriculum은 학습이 계속 진행되어도 성능이 추가로 향상되거나 유지되는 과적합에 대한 높은 회복력을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 단순한 모델 압축(Pruning)을 넘어, 용량의 '회복' 과정이 모델의 일반화 성능을 높일 수 있음을 보여주었다. 특히, 복잡한 Dynamic Update scheme보다 단순한 Identical Update scheme이 더 좋은 성과를 낸 점은 흥미롭다. 이는 모델 용량을 전략적으로 조절하는 행위 자체가 이미 강력한 정규화(Regularization) 효과를 제공하며, 이후의 학습은 표준적인 방식으로 진행해도 충분함을 시사한다.

### 한계 및 미해결 과제
1. **데이터셋의 한계**: WikiText2라는 단일 데이터셋에서 실험이 이루어졌으므로, 더 다양한 도메인의 데이터셋에서의 검증이 필요하다.
2. **학습률 스케줄러**: 논문에서는 고정된 스케줄러를 사용했으나, 최근 LLM에서 사용되는 정교한 학습률 스케줄러가 Cup Curriculum과 어떤 상호작용을 할지는 미지수이다.
3. **대규모 모델 적용**: Small/Medium/Large 모델을 사용했으나, 최신 초거대 언어 모델(LLM)에 적용했을 때 체크포인트 저장 비용 등을 실제로 얼마나 절감할 수 있을지는 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 NLP의 Transformer 모델에 대해 **용량을 줄였다가 다시 늘리는 'Cup Curriculum' 학습 전략**을 제안하였다. 실험 결과, 이 방식은 기존의 Early Stopping보다 **0.5%~2% 더 높은 성능**을 기록했으며, 특히 **과적합에 매우 강건**한 모습을 보였다. 이 연구는 향후 거대 모델의 학습 비용을 줄이면서도 성능을 최적화하는 새로운 학습 패러다임을 제시할 가능성이 크다.