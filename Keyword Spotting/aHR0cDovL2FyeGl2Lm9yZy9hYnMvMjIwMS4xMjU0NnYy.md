# PROGRESSIVE CONTINUAL LEARNING FOR SPOKEN KEYWORD SPOTTING

Yizheng Huang, Nana Hou, Nancy F. Chen (2022)

## 🧩 Problem to Solve

본 논문은 배포 후 새로운 키워드를 추가하여 모델을 업데이트해야 하는 Spoken Keyword Spotting (KWS) 시스템에서 발생하는 Catastrophic forgetting(치명적 망각) 문제를 해결하고자 한다.

KWS 모델은 일반적으로 낮은 연산량과 작은 메모리 점유율(small-footprint)을 위해 콤팩트한 모델 구조를 가진다. 이로 인해 소스 도메인 데이터로 학습된 모델이 런타임에 타겟 도메인의 새로운 키워드를 학습할 때, 기존에 학습했던 키워드에 대한 성능이 급격히 저하되는 문제가 발생한다. 기존의 Few-shot fine-tuning 방식은 타겟 도메인에 적응할 수는 있으나, 앞서 언급한 치명적 망각 문제가 심각하며 성능 보장을 위해 거대한 사전 학습 임베딩 모델이 필요하여 소형 KWS 시나리오에는 부적합하다는 한계가 있다. 따라서 본 연구의 목표는 모델 파라미터의 급격한 증가 없이 기존 지식을 보존하면서 새로운 키워드를 점진적으로 학습할 수 있는 효율적인 Continual Learning (CL) 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 소형 KWS를 위한 **Progressive Continual Learning (PCL-KWS)** 프레임워크를 제안한 것이다. 이 프레임워크의 중심 아이디어는 다음과 같다.

1. **Network Instantiator**: 새로운 학습 작업(Task)이 추가될 때마다 해당 작업 전용의 서브 네트워크(Sub-network)를 생성하여 기존 키워드에 대한 기억을 보존하고 새로운 키워드를 학습한다.
2. **Shared Memory**: 모든 서브 네트워크가 공유하는 메모리를 통해 이전 작업에서 학습된 지식을 활용함으로써 지식 전이(Knowledge Transfer)를 가능하게 하고 수렴 속도를 높인다.
3. **Keyword-aware Network Scaling**: 새로운 서브 네트워크를 생성할 때 키워드의 수에 따라 네트워크의 너비(Width)를 동적으로 조절하는 메커니즘을 도입하여, 성능은 유지하면서 모델 파라미터의 무분별한 증가를 억제한다.

## 📎 Related Works

논문에서는 기존의 Continual Learning 방법론을 크게 두 가지 범주로 나누어 설명하고 그 한계를 지적한다.

1. **Regularization-based Methods**: 이전 작업에서 학습된 중요 파라미터가 변경되지 않도록 손실 함수에 정규화 항을 추가하는 방식이다. 대표적으로 Elastic Weight Consolidation (EWC)와 Synaptic Intelligence (SI)가 있다. 이러한 방식은 추가 메모리가 필요 없으나, 정규화 제약이 너무 강할 경우 새로운 작업을 학습하는 능력이 떨어지거나(Low LA), 모델 크기가 고정되어 있어 학습 능력이 제한되는 한계가 있다.
2. **Replay-based Methods**: 과거의 데이터나 파라미터를 저장하는 버퍼를 사용하여 이전 작업을 재학습하는 방식이다. Naive Rehearsal (NR)은 과거 데이터를 저장하고, Gradient Episodic Memory (GEM)는 그래디언트의 방향을 제어하여 망각을 방지한다. 이 방식들은 망각 방지 성능은 뛰어나지만, 데이터 버퍼를 유지하기 위한 메모리 비용이 매우 크다는 단점이 있다.

PCL-KWS는 서브 네트워크를 점진적으로 확장하는 구조를 통해 Replay-based 방식의 메모리 버퍼 없이도 망각 문제를 해결하며, Regularization-based 방식보다 높은 학습 성능을 달성함으로써 기존 방식들과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

PCL-KWS는 입력 오디오에서 MFCC(Mel-Frequency Cepstral Coefficients) 특징을 추출한 후, 이를 Network Instantiator가 생성한 작업별 서브 네트워크에 통과시키는 구조를 가진다.

1. **학습 단계**: 새로운 작업 $\tau_t$가 들어오면, 이전의 $t-1$개 서브 네트워크들은 동결(Freeze)시킨다. Network Instantiator는 새로운 키워드 수에 맞는 $t$번째 서브 네트워크를 생성하여 학습시킨다.
2. **지식 공유**: 생성된 서브 네트워크는 Shared Memory에 접근하여 이전 작업들의 학습된 특징을 활용한다.
3. **추론 단계**: 실행 시점에 주어진 Task ID에 해당하는 서브 네트워크를 선택하여 평가를 수행한다.

### Keyword-aware Network Scaling

모델 파라미터의 증가를 억제하기 위해, 본 논문은 Dynamic Width Multiplier $\alpha$를 도입한다. 각 서브 네트워크의 채널 수에 곱해지는 $\alpha_{\tau}$는 다음과 같은 방정식으로 결정된다.

$$\alpha_{\tau} = \mu \frac{C_t}{C_0}, (\mu > 0)$$

여기서 $C_t$는 현재 학습하려는 작업 $\tau_t$의 키워드 수이고, $C_0$는 모델을 사전 학습(Pre-train)시킬 때 사용한 키워드 수이다. 일반적으로 증분 학습 단계에서 추가되는 키워드 수는 사전 학습 시의 키워드 수보다 적으므로 $\alpha \leq 1$이 되어 서브 네트워크의 크기가 효율적으로 작게 유지된다.

### 학습 목표 및 손실 함수

전체 학습 목표는 모든 $T$개의 작업에 대해 다음의 총 손실 함수 $L_{tot}$를 최소화하는 것이다.

$$L_{tot} = \sum_{t=0}^{T} \mathbb{E}_{(x_t, y_t) \sim \mathcal{D}_T} [L_{kws}(F_t(x_t; \theta_t), y_t)]$$

여기서 $L_{kws}$는 Cross-Entropy Loss를 의미하며, $F_t(x_t; \theta_t)$는 파라미터 $\theta_t$를 가진 $t$번째 KWS 모델이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Command (GSC) 데이터셋 (30개 영어 키워드 카테고리).
- **테스트 모델**: TC-ResNet-8 (Temporal Convolutional ResNet).
- **CL 시나리오**: 15개 키워드로 사전 학습 후, 3개씩의 새로운 키워드를 포함한 5개의 작업을 순차적으로 학습.
- **평가 지표**: Average Accuracy (ACC), Learning Accuracy (LA), Backward Transfer (BWT), Training Time (TT), Extra Param, Buffer Size.

### 주요 결과

실험 결과, PCL-KWS는 비교 대상 중 가장 우수한 성능을 보였다.

1. **정량적 성능**: PCL-KWS는 모든 작업에 대해 **92.8%의 평균 정확도(ACC)**를 기록하며 SOTA 성능을 달성하였다. 이는 Fine-tuning 베이스라인 대비 비약적인 향상이며, 가장 성능이 좋았던 Replay 방식인 NR보다도 8.7% 더 높은 정확도를 보인다.
2. **효율성**:
    - **메모리**: Replay-based 방법(NR, GEM)과 달리 버퍼가 전혀 필요 없다.
    - **시간**: NR 대비 학습 시간을 약 7배 단축시켰다.
    - **파라미터**: Keyword-aware scaling 덕분에 작업 수가 256개까지 늘어나더라도 추가 파라미터가 2M개 미만으로 유지되는 매우 효율적인 확장성을 보여주었다.
3. **망각 방지**: 작업 수가 증가함에 따라 SI나 EWC는 성능이 급격히 하락하는 반면, PCL-KWS는 높은 정확도를 일정하게 유지하여 Catastrophic forgetting에 매우 강함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 소형 KWS 모델에서 지속적 학습을 구현하기 위해 '구조적 확장'과 '동적 스케일링'이라는 두 가지 전략을 적절히 결합하였다.

**강점**으로는 기존의 정규화 기반 방식이 가진 '학습 능력 저하' 문제와 리플레이 기반 방식이 가진 '메모리 오버헤드' 문제를 동시에 해결했다는 점을 들 수 있다. 특히 $\alpha$ 값을 이용한 채널 스케일링은 모델의 유연성을 확보하면서도 하드웨어 제약이 심한 소형 기기 환경에서 실용적인 해결책이 될 수 있음을 보여준다.

**한계 및 논의사항**으로는, Task ID가 주어져야 해당 서브 네트워크를 선택할 수 있다는 점이다. 실제 환경에서 어떤 키워드 작업이 입력으로 들어왔는지 미리 알 수 없는 'Task-agnostic' 상황에서의 동작 여부는 본 논문에서 명시적으로 다루지 않았다. 또한, 서브 네트워크가 계속 추가됨에 따라 이론적으로는 파라미터가 계속 증가하므로, 매우 장기간의 학습 시에는 결국 메모리 한계에 도달할 가능성이 있다.

## 📌 TL;DR

본 논문은 소형 키워드 스포팅(KWS) 모델의 치명적 망각 문제를 해결하기 위해, 작업별 전용 서브 네트워크를 생성하고 키워드 수에 따라 크기를 조절하는 **PCL-KWS** 프레임워크를 제안한다. 이 방법은 데이터 버퍼 없이도 기존 지식을 완벽하게 보존하며, 새로운 키워드를 효율적으로 학습하여 92.8%의 평균 정확도를 달성하였다. 이는 메모리 제약이 엄격한 온디바이스 AI 환경에서 실시간으로 키워드를 추가해야 하는 시스템에 매우 중요한 적용 가능성을 제시한다.
