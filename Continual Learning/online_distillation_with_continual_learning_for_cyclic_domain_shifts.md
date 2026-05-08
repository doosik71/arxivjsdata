# Online Distillation with Continual Learning for Cyclic Domain Shifts

Joachim Houyon, Anthony Cioppa, Yasir Ghunaim, Motasem Alfarra, Anaïs Halin, Maxim Henry, Bernard Ghanem, Marc Van Droogenbroeck (2023)

## 🧩 Problem to Solve

본 논문은 실시간 딥러닝 네트워크를 빠르게 적응시키기 위한 Online Distillation 과정에서 발생하는 **파괴적 망각(Catastrophic Forgetting)** 문제를 해결하고자 한다.

일반적으로 Online Distillation은 느리지만 정확한 Teacher 모델을 사용하여 실시간으로 Student 모델을 업데이트하는 방식이다. 그러나 데이터의 분포가 변하는 Domain Shift가 발생할 때, Student 모델이 새로운 도메인의 데이터로 업데이트되면서 이전에 학습했던 도메인의 지식을 잊어버리는 현상이 발생한다. 특히, 자율주행 환경처럼 고속도로(Highway)와 도심(Downtown) 환경이 반복적으로 나타나는 **순환적 도메인 시프트(Cyclic Domain Shifts)** 상황에서는 모델이 이전 도메인으로 돌아왔을 때 다시 낮은 성능을 보이는 문제가 심각하게 나타난다.

따라서 본 연구의 목표는 Continual Learning(CL) 기법을 Online Distillation 프레임워크에 통합하여, 도메인 시프트 상황에서도 이전 지식을 유지하면서 새로운 도메인에 효과적으로 적응하는 강건한 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **순환적 온라인 지속 학습(Cyclic Online Continual Learning) 문제 정의**: 데이터 스트림이 두 가지 이상의 분포 사이를 반복적으로 오가는 구체적인 시나리오를 설정하고, 이를 평가하기 위한 적절한 지표(BWT, FWT 등)를 제안하였다.
2. **Online Distillation과 CL 기법의 통합**: 기존의 실시간 Online Distillation 프레임워크에 **Regularization-based** 및 **Replay-based** 지속 학습 방법론을 결합하여 파괴적 망각을 완화하는 구조를 설계하였다.
3. **광범위한 벤치마크 분석**: 다양한 리플레이 전략(FIFO, Uniform, Prioritized, MIR)과 정규화 기법(ER-ACE, LwF, MAS, RWalk)을 실험하여, 순환적 도메인 시프트 상황에서 어떤 조합이 가장 효과적인지 정량적/정성적으로 분석하였다.

## 📎 Related Works

### Domain Shifts

데이터의 통계적 분포가 변하는 Domain Shift는 자율주행과 같은 오픈 월드 시나리오에서 빈번하게 발생한다. 조명, 날씨, 도로 환경의 변화 등이 주요 원인이며, 이를 해결하기 위한 Domain Adaptation 연구들이 진행되어 왔으나, 본 논문은 특히 '순환적'으로 발생하는 시프트에 집중한다.

### Online Distillation

실시간 시스템에서는 속도, 성능, 일반화 능력 사이의 트레이드오프가 존재한다. 기존 연구(Cioppa et al. [10])는 무거운 Teacher 모델이 생성한 Pseudo Ground-Truth를 통해 가벼운 Student 모델을 온라인으로 학습시키는 방식을 제안하였다. 하지만 이 방식은 현재 도메인에 과하게 특화되어 이전 도메인에 대한 지식을 빠르게 잊는다는 한계가 있다.

### Continual Learning (CL)

CL은 새로운 데이터를 학습하면서도 이전 지식을 유지하는 것을 목표로 한다. 주요 접근법으로는 중요 파라미터의 변화를 제한하는 Regularization-based 방법과 과거 데이터를 저장해두고 다시 학습하는 Replay-based 방법이 있다. 기존 CL 연구들은 주로 레이블이 있는 지도 학습 환경을 가정하지만, 본 논문은 레이블이 없는 Unsupervised 설정의 Online Distillation에 이를 적용하였다.

## 🛠️ Methodology

### 1. Online Distillation Framework

본 논문은 Fast Route(추론)와 Slow Route(학습)로 구성된 학생-교사 구조를 사용한다.

- **Fast Route**: Student 네트워크 $S$가 실시간으로 프레임 $x_i$에 대해 예측값 $\hat{y}_i = S(x_i)$를 생성한다.
- **Slow Route**: Frozen 상태의 Teacher 네트워크 $T$가 비동기적으로 의사 정답(Pseudo Ground-Truth) $\tilde{y}_i' = T(x_i')$를 생성한다.
- **학습 절차**: $(x_i', \tilde{y}_i')$ 쌍은 온라인 데이터셋 $D$에 저장되며, Student의 복사본인 $S^c$는 $D$에서 샘플링된 데이터를 통해 다음과 같은 손실 함수를 최소화하며 학습된다.

$$L = \sum_{n=1}^{N} L(S^c(x_n), \tilde{y}_n)$$

학습이 완료된 $S^c$의 파라미터 $\theta$는 주기적으로 $S$에 복사된다.

### 2. Replay-based Methods ($CL_{Rep}$)

리플레이 버퍼 $D$의 업데이트 함수 $f^U$와 샘플링 함수 $f^S$를 수정하여 망각을 줄인다. 버퍼 크기를 $M$으로 설정하고 매 스텝 $N$개의 샘플을 선택한다.

- **FIFO**: 가장 오래된 데이터를 삭제하고 최신 데이터를 저장하는 기본 방식이다.
- **Uniform**: 버퍼의 임의 위치에 데이터를 저장하고 무작위로 샘플링한다.
- **Prioritized**: 손실 값 $L(S(x_n), T(x_n))$을 기반으로 중요도 점수 $I_n$을 부여하며, 점수가 낮은(이미 잘 학습된) 데이터를 우선적으로 삭제한다.
- **MIR (Maximally Interfered Retrieval)**: 현재 들어온 데이터로 인해 파라미터 업데이트 시 가장 큰 간섭(Interference)을 받는 샘플들을 버퍼에서 선택하여 학습한다.

### 3. Regularization-based Methods ($CL_{Reg}$)

손실 함수에 정규화 항 $R$을 추가하여 중요 파라미터의 급격한 변화를 억제한다.

$$L = \sum_{n=1}^{N} L(S^c(x_n), \tilde{y}_n) + R$$

- **ER-ACE**: 리플레이 버퍼 사용 시, 새로운 데이터에 대해서는 현재 클래스에 대해서만 손실을 계산하는 비대칭 업데이트를 수행한다.
- **LwF (Learning without Forgetting)**: 이전 버전의 Student 네트워크를 유지하며, 현재 출력값이 이전 모델의 출력값과 유사하도록 지식 증류(Knowledge Distillation)를 적용한다.
- **MAS (Memory Aware Synapses)**: 각 파라미터의 중요도를 계산하여, 중요한 파라미터가 크게 변할 때 페널티를 부여한다.
- **RWalk**: EWC와 PI 기법을 일반화한 형태로, 파라미터의 중요도 점수를 기반으로 정규화를 수행한다.

### 4. Evaluation Metrics

- **mIoU**: 현재 도메인에서의 세그멘테이션 성능을 측정한다.
- **Backward Transfer (BWT)**: 현재 모델이 이전 도메인의 데이터 $X_{i'-h}$에 대해 내는 성능을 측정하여 망각 정도를 평가한다.
- **Forward Transfer (FWT)**: 현재 모델이 아직 보지 않은 미래 도메인의 데이터 $X_{i'+h}$에 대해 내는 성능을 측정한다.
- **Final BWT**: 학습이 모두 끝난 최종 모델을 전체 스트림에 대해 평가한다.

## 📊 Results

### 실험 설정

- **태스크**: 자율주행 도로의 Semantic Segmentation.
- **데이터셋**: 고속도로(Highway)와 도심(Downtown) 환경의 비디오 클립을 교차 결합하여 순환적 도메인 시프트를 시뮬레이션하였다.
- **모델**: Teacher는 SegFormer, Student는 TinyNet을 사용하였다.
- **지표**: Pseudo Ground-Truth 대비 mIoU를 측정하였다.

### 정량적 결과 (Table 1 분석)

- **Memoryless vs Baseline**: 데이터를 저장하지 않는 Memoryless 방식보다 리플레이 버퍼를 사용하는 Baseline(FIFO)이 모든 지표에서 우수하였다.
- **Replay-based의 효과**: Uniform, Prioritized, MIR 방식 모두 Baseline보다 높은 성능을 보였으며, 특히 **MIR**이 전반적으로 가장 우수한 성능을 기록하였다.
- **Regularizer의 영향**: MAS와 LwF는 오히려 성능을 저하시켰다. 이는 이 기법들이 모델의 유연성(Plasticity)을 너무 억제하여 온라인 환경에서의 빠른 적응을 방해했기 때문으로 분석된다. 반면, **ACE와 RWalk**는 성능을 소폭 향상시켰다.
- **최적 조합**: **MIR + RWalk** 조합이 mIoU 및 BWT, Final BWT 측면에서 매우 뛰어난 성능을 보였다.

### 정성적 결과 및 분석

- **망각의 가시화**: Figure 3의 mIoU 그래프를 보면, Baseline은 도메인이 다시 돌아왔을 때(두 번째 사이클부터) 성능이 급격히 떨어지는 반면, MIR+RWalk는 높은 성능을 유지한다.
- **시각적 분석**: Figure 4의 세그멘테이션 마스크 결과, Baseline은 도메인 전환 직후 매우 불완전한 마스크를 생성하지만, MIR 및 MIR+RWalk는 Ground Truth에 가까운 정교한 결과를 보여준다.

## 🧠 Insights & Discussion

### 강점

본 연구는 실시간 Online Distillation이 가진 치명적인 약점인 '파괴적 망각'을 CL 기법의 통합이라는 실용적인 방법으로 해결하였다. 특히, 단순히 최신 논문의 기법들을 적용한 것에 그치지 않고, 온라인 스트림이라는 특수성에 맞춰 Warm-up 기간과 업데이트 주기(Update Frequency)를 설정하여 적용한 점이 돋보인다.

### 한계 및 해석

- **정규화 기법의 양면성**: MAS나 LwF 같은 전통적인 CL 기법들이 온라인 적응 성능을 떨어뜨렸다는 점은 매우 흥미로운 통찰이다. 이는 '안정성(Stability)'과 '가소성(Plasticity)' 사이의 트레이드오프가 온라인 학습 환경에서는 일반적인 배치 학습 환경과 다르게 작용함을 시사한다.
- **가정 사항**: 본 논문은 Teacher 모델이 항상 신뢰할 수 있는 Pseudo-labels를 제공한다는 가정을 전제로 한다. 만약 Teacher 모델 자체가 새로운 도메인에서 낮은 성능을 보인다면, Student 모델은 잘못된 정보를 학습하게 될 위험이 있다.

### 비판적 논의

실험에서 사용된 도메인이 고속도로와 도심 두 가지로 제한적이다. 실제 환경에서는 훨씬 더 다양하고 연속적인 도메인 변화가 발생하므로, 더 많은 수의 도메인이 복잡하게 얽힌 환경에서도 MIR+RWalk 조합이 동일한 효율성을 보일지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 실시간 온라인 지식 증류(Online Distillation) 과정에서 발생하는 파괴적 망각 문제를 해결하기 위해 **지속 학습(Continual Learning) 기법을 통합**한 프레임워크를 제안하였다. 특히 **MIR(Maximally Interfered Retrieval)** 기반의 리플레이 전략과 **RWalk** 정규화의 조합이 순환적 도메인 시프트 환경에서 가장 효과적임을 입증하였다. 이 연구는 자율주행과 같이 환경 변화가 심한 실시간 시스템에서 모델의 강건성과 정확도를 동시에 확보할 수 있는 실무적인 방향성을 제시한다.
