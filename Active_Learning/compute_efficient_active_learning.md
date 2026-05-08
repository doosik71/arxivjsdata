# Compute-Efficient Active Learning

Gábor Németh, Tamás Matuszka (2023)

## 🧩 Problem to Solve

본 논문은 대규모 데이터셋을 대상으로 하는 Active Learning(능동 학습) 과정에서 발생하는 막대한 계산 비용 문제를 해결하고자 한다.

Active Learning은 모델 학습에 가장 유용한(informative) 샘플만을 선택적으로 라벨링하여 전체 라벨링 비용을 줄이는 강력한 패러다임이다. 그러나 전통적인 Active Learning 방식은 매 반복(iteration)마다 전체 미라벨링 데이터셋($D_{\text{unlabeled}}$)에 대해 Acquisition Function(획득 함수)을 계산해야 한다. 데이터셋의 규모가 거대해질수록 모든 샘플의 중요도를 평가하는 과정에서 발생하는 계산 부하가 기하급수적으로 증가하며, 이는 실질적인 확장성과 효율성을 저해하는 결정적인 병목 구간이 된다.

따라서 본 연구의 목표는 모델의 성능을 유지하거나 오히려 향상시키면서도, 계산 자원 소모를 획기적으로 줄일 수 있는 방법론적으로 범용적인(method-agnostic) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"Acquisition Function의 과거 값은 미래 값을 예측하는 좋은 지표가 된다"**는 가설에 기반한다. 즉, 특정 샘플에 대해 모델이 이미 확신(certainty)을 가졌다면, 이후의 학습 단계에서도 그 상태가 급격히 변할 가능성이 낮다는 직관을 이용한 것이다.

이를 위해 저자들은 모든 샘플을 매번 평가하는 대신, 과거의 평가 값을 바탕으로 확률적으로 샘플을 선택하여 소규모의 **Candidate Pool(후보군 풀)**을 구성하고, 이 풀에 대해서만 Acquisition Function을 업데이트하는 전략을 제안한다. 이는 계산 효율성을 극대화함과 동시에, 무작위 샘플링과 정보 기반 샘플링 사이의 적절한 보간(interpolation) 역할을 수행하여 모델 성능을 최적화한다.

## 📎 Related Works

기존의 효율적인 Active Learning 연구들은 주로 개별 샘플의 Epistemic Uncertainty(인식론적 불확실성)를 계산하는 비용을 줄이는 데 집중해 왔다.

1. **Bayesian Approximation:** Model Ensembling이나 Monte Carlo Dropout을 사용하여 사후 분포를 근사하지만, 여러 번의 Forward Pass가 필요하여 대규모 데이터셋에서는 계산 비용이 매우 높다.
2. **Efficient Metrics:** Virtual Adversarial Active Learning(VirAAL)의 LDR 메트릭, 별도의 네트워크를 이용한 Pretext Task 기반 선택, VAAL의 Adversarial Training, 그리고 Learning Loss 예측 헤드를 추가하는 방식 등이 제안되었다.
3. **Evidential Deep Learning:** 최근 HUA(Hierarchical Uncertainty Aggregation)와 같은 프레임워크가 객체 탐지 분야에서 효율적인 정보량 계산을 가능케 하였다.

이러한 기존 방식들은 단일 샘플의 평가 비용을 낮추는 데는 기여했지만, 여전히 **모든 미라벨링 데이터 포인트에 대해 평가를 수행해야 한다**는 근본적인 계산 부담을 안고 있다. 반면, 본 논문의 제안 방식은 평가 대상 자체를 전략적으로 줄임으로써 기존의 효율적인 Acquisition Function들과 상호 보완적으로 결합하여 시너지 효과를 낼 수 있다는 점에서 차별점을 가진다.

## 🛠️ Methodology

본 논문은 과거의 Acquisition Function 평가 값을 활용하여 전략적으로 부분 샘플링(subsampling)하는 기법을 제안한다. 전체 프로세스는 다음과 같은 알고리즘 흐름을 따른다.

### 1. 전체 파이프라인 (Algorithm 1)

1. **초기화:** 전체 미라벨링 데이터셋 $D_{\text{unlabeled}}$에 대해 초기 Acquisition Function 값을 계산한다.
2. **반복 루프 ($T$회 수행):**
    - 현재 라벨링된 데이터 $D_{\text{labeled}}$로 모델 $M$을 학습시킨다.
    - 저장된 Acquisition Values에 $\text{softmax}$를 적용하여 각 샘플이 선택될 확률 $P$를 계산한다:
      $$P = \text{softmax}(\text{AcquisitionValues})$$
    - 확률 $P$에 기반하여 전체 데이터 수 $N$의 $\alpha$ 비율만큼의 샘플을 추출하여 **Candidate Pool**을 구성한다.
    - **오직 Candidate Pool에 속한 샘플들에 대해서만** 현재 모델 $M$을 이용해 Acquisition Function 값을 업데이트한다.
    - 업데이트된 값 중 상위 $K$개의 샘플($X_{\text{label}}$)을 선택하여 라벨링을 진행하고 $D_{\text{labeled}}$에 추가한다.
    - 선택된 샘플을 $D_{\text{unlabeled}}$에서 제거한다.

### 2. 주요 구성 요소 및 특징

- **범용성 (Method-Agnostic):** 특정 함수에 종속되지 않으며 Shannon Entropy, Variation Ratios ($\text{varR}$), Ensemble Score, BALD 등 다양한 Acquisition Function과 결합 가능하다.
- **회귀 문제 적용:** 회귀 문제의 경우 출력 분포의 편차(deviation)를 측정하여 Acquisition Function 값으로 사용할 수 있음을 명시하였다.
- **데이터 제외 전략:** Acquisition Function 값이 극도로 낮은 샘플들을 미라벨링 데이터셋에서 완전히 제거함으로써 계산 및 저장 공간의 효율성을 추가로 높일 수 있다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** MNIST, CIFAR-10
- **모델 아키텍처:** MNIST는 단순 CNN, CIFAR-10은 VGG-11 (Batch Normalization 적용)
- **지표:** Test Accuracy (분류 정확도)
- **비교 대상 (Baselines):**
  - Random Sampling (전체 풀에서 무작위 선택)
  - Full-set Acquisition Function (전체 풀에서 Entropy 또는 $\text{varR}$ 기반 선택)
- **하이퍼파라미터:** 반복 횟수 $T=10$, CIFAR-10의 경우 초기 풀 크기(1%, 25%) 및 총 획득 크기(10%, 25%)를 변경하며 실험.

### 2. 주요 결과

- **정성적/정량적 성능:**
  - 모든 실험에서 제안 방법이 Random Baseline보다 뛰어난 성능을 보였다.
  - 일부 설정에서는 전체 데이터를 평가한 Entropy/$\text{varR}$ 기반 방식보다 더 높은 정확도를 기록하였다.
  - 성능 차이가 미미하더라도 계산 비용 면에서 압도적인 이점을 가짐을 확인하였다.
- **계산 효율성:**
  - CIFAR-10 실험에서 전체 데이터의 26%만을 사용하여 학습했을 때, Baseline 대비 학습 시간을 최대 **25% 절감**하였다 (672분 $\rightarrow$ 502분, NVIDIA GeForce GTX TITAN X 기준).
  - 데이터셋 규모가 커질수록 Candidate Pool의 비율이 상대적으로 낮아지므로, 실제 환경에서는 런타임 감소 효과가 더욱 극대화될 것으로 분석된다.
- **범용성 검증:** aiMotive Multimodal Dataset을 이용한 3D 객체 탐지(BEVFusion 기반 모델) 예비 실험에서도 유사한 성능 향상 및 효율성 증대 현상이 관찰되었다.

## 🧠 Insights & Discussion

### 1. 성능 향상의 원인 분석

제안 방법이 단순히 계산량을 줄이는 것을 넘어 성능을 향상시킨 이유는 **"무작위 샘플링과 정보 기반 샘플링의 보간(interpolation)"** 효과로 해석할 수 있다.
전통적인 Acquisition Function은 모델이 불확실해하는 샘플에만 집중하여 다양성이 낮은 샘플들을 선택하는 경향이 있다. 하지만 본 방법은 $\text{softmax}$ 확률 기반의 샘플링을 통해 약간의 무작위성을 부여함으로써, 결과적으로 더 다양하고 유용한 샘플들을 선택하게 된다.

### 2. Cold Start 문제 해결

Active Learning의 고질적인 문제인 'Cold Start'(초기 라벨링 데이터가 너무 적어 Acquisition Function이 제대로 작동하지 않는 현상) 상황에서, 본 방법의 확률적 샘플링이 초기 단계의 무작위 샘플링과 유사한 역할을 수행하여 이를 효과적으로 완화한다.

### 3. 한계 및 논의사항

- 본 논문은 벤치마크 데이터셋을 중심으로 검증되었으나, 실제 대규모 환경에서의 정확한 정량적 수치에 대해서는 추가 실험이 필요함을 언급하고 있다.
- $\alpha$(Subsample ratio)와 $\text{softmax}$의 Temperature 파라미터가 성능에 영향을 미칠 수 있으나, 이에 대한 최적화 가이드라인은 명시되지 않았다.

## 📌 TL;DR

본 연구는 대규모 데이터셋의 Active Learning에서 발생하는 계산 병목을 해결하기 위해, **과거의 중요도 값을 활용해 소규모의 Candidate Pool을 구성하고 이들만 집중적으로 평가하는 효율적인 프레임워크**를 제안한다. 실험 결과, 계산 비용을 획기적으로 줄이면서도 무작위성과 정보성을 동시에 확보함으로써 기존의 전체 평가 방식보다 우수하거나 대등한 모델 성능을 달성하였다. 이 방법은 모델이나 태스크에 구애받지 않는 범용적인 구조이므로, 향후 자율주행과 같은 초거대 데이터셋 기반의 딥러닝 학습 파이프라인에 적용되어 학습 효율을 크게 높일 가능성이 크다.
