# Rainbow Keywords: Efficient Incremental Learning for Online Spoken Keyword Spotting

Yang Xiao, Nana Hou, Eng Siong Chng (2022)

## 🧩 Problem to Solve

본 논문은 배포 후 새로운 키워드를 추가해야 하는 온라인 Keyword Spotting (KWS) 시스템에서 발생하는 Catastrophic Forgetting(치명적 망각) 문제를 해결하고자 한다. 특히, KWS 모델이 메모리 자원이 매우 제한적인 edge device에서 동작해야 한다는 점에서 다음과 같은 세부 문제들을 다룬다.

첫째, 새로운 타겟 도메인의 키워드를 학습시키기 위해 Few-shot fine-tuning을 적용할 경우, 기존에 학습했던 소스 도메인의 지식을 잃어버리는 현상이 발생한다. 둘째, 기존의 Progressive Continual Learning 방식은 각 태스크를 구분하기 위한 Task-ID 정보가 필요하며, 태스크 수가 증가함에 따라 모델의 저장 공간이 함께 증가하여 edge device에 적용하기 어렵다. 

따라서 본 연구의 목표는 Task-ID 정보 없이도 적은 수의 파라미터와 메모리를 사용하여, 이전 지식을 유지하면서 새로운 키워드를 효율적으로 학습할 수 있는 Incremental Learning 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Diversity-aware Incremental Learning**이다. 제한된 메모리 내에서 효율적으로 지식을 보존하기 위해, 단순히 데이터를 저장하는 것이 아니라 '다양성'이 높은 샘플을 선택하여 저장하고 이를 활용해 학습하는 방식을 제안한다.

주요 기여 사항은 다음과 같다.
1. **Diversity-aware Sampler**: 분류 불확실성(Classification Uncertainty)을 계산하여 과거 및 현재 데이터 중 가장 다양성이 높은 샘플을 선택함으로써 메모리 효율성을 극대화한다.
2. **Mixed-labeled Data Augmentation**: 선택된 소수의 샘플에 대해 Mixup 기반의 데이터 증강을 적용하여 학습 데이터의 다양성을 추가로 확보한다.
3. **Knowledge Distillation (KD) Loss**: 이전 태스크의 모델을 Teacher 모델로, 현재 모델을 Student 모델로 설정하여 지식을 전수함으로써 치명적 망각을 방지한다.

## 📎 Related Works

논문에서는 KWS 모델의 적응을 위해 Few-shot fine-tuning과 Progressive Continual Learning 두 가지 접근 방식을 언급한다. Few-shot fine-tuning은 새로운 시나리오에 빠르게 적응할 수 있으나, 이전 데이터에 대한 성능이 급격히 떨어지는 Catastrophic Forgetting 문제가 발생한다. 

반면, Progressive Continual Learning은 태스크별 서브 네트워크를 통해 망각을 방지하지만, 실무에서 항상 제공되기 어려운 Task-ID가 필수적이며, 태스크가 늘어날수록 모델 크기가 커져 edge device의 저장 용량 한계를 초과한다는 한계가 있다. 제안된 Rainbow Keywords (RK) 방식은 이러한 Task-ID 의존성을 제거하고 고정된 모델 크기를 유지하면서도 성능을 확보한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인
RK의 전체 워크플로우는 다음과 같다. 먼저 현재 태스크의 입력 데이터 $D_t^S$와 이전 태스크의 저장된 예시 $D_{t-1}^E$를 Diversity-aware Sampler에 입력한다. 샘플러는 불확실성 계산을 통해 다양성이 높은 샘플 $D_t^E$를 선택하여 저장소에 보관한다. 이후 데이터 증강을 통해 다양성을 높이고, Knowledge Distillation 손실 함수를 사용하여 Student 모델을 업데이트한다.

### 2. Diversity-Aware Sampler
메모리 효율을 위해 특징 공간(Feature space) 내에서 상대적 위치가 다양한 샘플을 선택한다. 이는 Monte-Carlo (MC) 방법을 통한 분류 불확실성 추정으로 구현된다.

- **불확실성 추정**: 각 샘플 $x$에 대해 5가지 섭동(Perturbations: Clipping Distortion, TimeMask, Shift, PitchShift, FrequencyMask)을 가한 $\hat{x}$들을 생성하고, 예측값의 일관성을 통해 불확실성 $u(x)$를 계산한다.
$$u(x) \approx 1 - \frac{1}{K} \sum_{k=1}^{K} P(y=c|\hat{x}^k)$$
여기서 $K$는 5가지 섭동 전략을 의미하며, $u(x)$ 값이 클수록 모델이 해당 샘플의 섭동에 대해 확신이 없음을 의미하며, 이는 곧 해당 샘플이 특징 공간에서 다양성을 가진다고 판단하는 근거가 된다.
- **샘플 선택**: 계산된 $u(x)$를 기준으로 내림차순 정렬하여 상위 $L$개의 샘플을 선택해 $D_t^E$에 저장한다.

### 3. Data Augmentation
제한된 메모리로 인해 저장된 샘플 수가 매우 적으므로, 추가 저장 공간 없이 데이터를 늘리기 위해 두 개의 오디오 발화를 무작위로 섞는 Mixup 방식을 적용하여 학습 데이터의 다양성을 높인다.

### 4. Knowledge Distillation (KD) Loss
이전 태스크의 모델(Teacher)이 가진 지식을 현재 모델(Student)에게 전달하여 망각을 방지한다.

- **KD Softmax**: 온도 하이퍼파라미터 $T=2.0$을 적용하여 소프트맥스 값을 계산한다.
$$\sigma(o_i^t(x); N_{t-1}) = \frac{\exp(o_i^t / T)}{\sum_{j=1}^{N_{t-1}} \exp(o_j^t / T)}$$
- **KD Loss**: Teacher 모델과 Student 모델의 출력 로짓(logits) 간의 차이를 최소화한다.
$$L^{KD}(o_t(x), o_{t-1}(x)) = \sum_{i=1}^{N_{t-1}} \sigma(o_i^t(x)) \log \sigma(o_{t-1}^i(x))$$
- **전체 손실 함수**: Cross-Entropy 손실($L^{CE}$)과 $L^{KD}$를 가중 합산하여 최종 최적화를 수행한다.
$$L^{total}(x,y) = \lambda L^{CE}(x,y) + (1-\lambda) L^{KD}(o_t(x), o_{t-1}(x))$$
이때 가중치 $\lambda$는 $\sqrt{1 - \frac{N_{t-1}}{N_t}}$로 설정된다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Google Speech Command (GSC) v1 (30개 키워드, 64,727개 클립).
- **모델**: TC-ResNet-8 (1D-Conv 및 3개의 Residual block으로 구성).
- **평가 지표**: Average Accuracy (ACC), Backward Transfer (BWT), 파라미터 수, 메모리 사용량.
- **비교 대상**: Fine-tune, NR, iCaRL, EWC, RWalk, BiC, PCL-KWS, Joint training.

### 2. 주요 결과
- **KD Loss의 효과**: KD Loss를 적용했을 때 BWT(과거 태스크 성능 유지 정도)가 상대적으로 54.5% 향상되었다.
- **메모리 크기에 따른 성능**: 메모리 크기 $L$이 300 또는 500으로 매우 제한적인 상황에서도 RK 방식이 다른 베이스라인들보다 높은 ACC를 기록하였다.
- **종합 성능 비교**: 태스크 수가 5일 때, RK-3000(메모리 크기 3000)은 가장 우수한 베이스라인인 RWalk 대비 Average Accuracy에서 4.2%의 절대적 성능 향상을 보였으며, 이는 상한선인 Joint training 성능에 근접한 결과이다. 
- **효율성**: RK-500은 매우 적은 메모리(16.2M)를 사용하면서도 다른 무거운 베이스라인들과 대등한 성능을 보여 edge device에 매우 적합함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 제한된 메모리 환경에서 **'어떤 데이터를 저장할 것인가'**에 집중하여, 분류 불확실성이 높은 샘플을 선택하는 것이 단순 무작위 선택이나 평균값 중심 선택보다 효과적임을 보여주었다. 이는 불확실성이 높은 샘플이 클래스의 경계나 특이 케이스를 포함하고 있어, 적은 수의 샘플만으로도 클래스의 대표성을 유지하는 데 유리하기 때문으로 해석된다.

또한, $\lambda$ 값을 태스크 수의 비율에 따라 동적으로 조절함으로써 새로운 지식 습득과 기존 지식 보존 사이의 균형을 맞춘 점이 긍정적이다. 다만, 제안된 방식이 매우 다양한 환경(소음, 화자 변화 등)에서도 동일한 다양성 샘플링 효과를 거둘 수 있는지에 대한 추가적인 분석은 명시되지 않았다.

## 📌 TL;DR

이 논문은 edge device를 위한 효율적인 KWS 증분 학습 방법론인 **Rainbow Keywords (RK)**를 제안한다. 불확실성 기반의 다양성 샘플러, Mixup 데이터 증강, 그리고 지식 증류(KD) 손실 함수를 결합하여, Task-ID 없이도 메모리 사용량을 최소화하면서 치명적 망각을 효과적으로 억제한다. 실험 결과, 기존 SOTA 모델 대비 메모리 효율성을 크게 높이면서도 평균 정확도를 4.2% 향상시켜 실제 저전력 기기의 온라인 키워드 업데이트 시스템에 적용될 가능성이 높다.