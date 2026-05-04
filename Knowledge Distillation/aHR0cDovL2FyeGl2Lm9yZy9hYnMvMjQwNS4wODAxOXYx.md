# AdaKD: Dynamic Knowledge Distillation of ASR models using Adaptive Loss Weighting

Shreyan Ganguly, Roshan Nayak, Rakshith Rao, Ujan Deb, Prathosh AP (2024)

## 🧩 Problem to Solve

본 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 모델의 크기를 줄이면서도 성능을 유지하는 모델 압축 문제에 집중한다. 최신 ASR 모델인 Whisper나 wav2vec 2.0은 수억 개에서 수십억 개의 파라미터를 가진 거대 모델로, 높은 추론 지연 시간(Inference Latency)으로 인해 CPU 기반의 실시간 전사(Transcription)나 에지 디바이스(Edge Device)에서의 사용이 어렵다는 한계가 있다.

이를 해결하기 위해 지식 증류(Knowledge Distillation, KD) 기법이 사용되지만, 기존의 방식들은 대개 태스크 전용 손실 함수(Task-specific loss)와 증류 손실 함수(Distillation loss)에 고정된 가중치를 부여한다. 저자들은 이러한 정적 가중치 방식이 각 학습 샘플이 가진 서로 다른 난이도를 반영하지 못하며, 결과적으로 최적의 성능을 달성하는 데 방해가 된다는 점을 지적한다. 따라서 본 논문의 목표는 샘플별 난이도에 따라 손실 함수의 가중치를 동적으로 조정하는 Adaptive Knowledge Distillation(AdaKD) 기법을 제안하여 모델 압축 효율을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 커리큘럼 학습(Curriculum Learning)의 개념을 지식 증류에 도입하여, 학습 과정에서 쉬운 샘플부터 어려운 샘플 순으로 지식을 전달하는 것이다.

구체적으로, 교사 모델(Teacher model)이 느끼는 손실 값(Teacher loss)을 샘플의 난이도를 측정하는 척도로 활용한다. 학습 초기에는 교사 모델의 손실이 낮은 '쉬운 샘플'에 더 많은 가중치를 두어 증류를 진행하고, 학습이 진행됨에 따라 점진적으로 교사 모델의 손실이 높은 '어려운 샘플'의 비중을 높이는 동적 가중치 전략을 사용한다. 이는 어려운 샘플들이 클래스 간의 복잡한 관계에 대한 풍부한 숨겨진 지식(Hidden knowledge)을 담고 있다는 직관에 기반하며, 쉬운 지식을 먼저 습득한 후 어려운 지식을 배우는 것이 효율적이라는 설계 의도를 가지고 있다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 다룬다.

- **Knowledge Distillation (KD):** Hinton 등에 의해 제안된 기법으로, 거대 모델(Teacher)의 출력 로짓(Logits)을 작은 모델(Student)이 모방하게 함으로써 성능을 전이시킨다. 로짓 증류(Logit distillation)와 특징 증류(Feature distillation)로 나뉘며, 최근에는 구조적 지식을 전이하는 Relational KD 등이 연구되었다.
- **KD for ASR:** wav2vec 2.0 모델의 파라미터를 획기적으로 줄이려는 시도들이 있었으며, 최근 Whisper 모델에 대해서도 vanilla KD를 적용한 연구가 진행되었다.
- **Curriculum Learning:** 인간의 학습 방식에서 영감을 받아 샘플을 난이도 순으로 제시하는 전략이다. 최근에는 confidence-aware 손실 함수나 샘플의 복잡도에 따른 시퀀싱 연구가 진행되었다.

기존의 KD 방식들이 모든 샘플에 동일한 중요도를 부여하거나, 단순히 온도(Temperature) 파라미터를 조절하는 수준이었다면, 제안 방법론은 교사 모델의 손실을 기반으로 인스턴스 수준(Instance-level)에서 가중치를 동적으로 제어한다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인

AdaKD는 기존의 어떤 태스크 전용 손실 함수나 증류 목적 함수와도 결합할 수 있는 plug-and-play 구조를 가진다. 전체적인 흐름은 교사 모델의 손실을 계산하고 $\rightarrow$ 이를 바탕으로 난이도 계수를 산출하며 $\rightarrow$ 최종적으로 학생 모델의 학습에 사용할 동적 가중치 $\alpha$를 결정하는 순으로 진행된다.

### 상세 방법론 및 방정식

**1. 기본 지식 증류 손실**
학생 모델은 교사 모델의 출력 로짓을 모방하기 위해 Kullback–Leibler (KL) divergence를 사용한다.
$$L_{kd}(y_s, y_t) = \tau^2 KL(y_s, y_t)$$
여기서 $y_s$와 $y_t$는 각각 학생과 교사 모델의 출력이며, $\tau$는 temperature 하이퍼파라미터이다.

**2. 최종 결합 손실 함수**
최종 손실 함수 $L_{student}$는 태스크 전용 손실 $L_{ts}$와 증류 손실 $L_{kd}$의 가중 합으로 정의된다.
$$L_{student} = (1-\alpha)L_{ts}(y_i, x_i | \theta) + \alpha L_{kd}(y_{si}, y_{ti})$$
여기서 $\alpha$는 증류 목적 함수의 중요도를 결정하는 가중치이다.

**3. AdaKD의 동적 가중치 계산**
$\alpha$는 샘플의 난이도 계수 $d_f$에 의해 결정된다.
$$\alpha = e^{-\frac{1}{\sqrt{d_f}}}$$
이때 난이도 계수 $d_f$는 다음과 같이 계산된다.
$$d_f = e^{-k(x-t)}$$

- $x$: 교사 모델의 해당 샘플에 대한 손실 값($L_{ts}(y, x | \theta_t)$).
- $t$: 난이도의 기준점이 되는 임계값으로, 일반적으로 전체 훈련 세트의 교사 모델 손실 평균값으로 설정한다.
$$t = \frac{1}{N} \sum_{i=1}^{N} L_{ts}(y_i, x_i | \theta)$$
- $k$: 증류 손실의 중요도를 조절하는 하이퍼파라미터이며, 학습 과정 동안 선형적으로 감소한다.

**4. 학습 절차**

- 초기 단계에서는 $k$ 값을 크게 설정하여, 교사 손실이 높은(어려운) 샘플의 $\alpha$ 값을 최소화(약 0.1)하고 쉬운 샘플부터 학습시킨다.
- 학습이 진행됨에 따라 $k$ 값을 감소시켜, 점차 어려운 샘플들에 대해서도 $\alpha$ 가중치가 높아지도록 유도한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Common Voice 11 (힌디어, 중국어, 타밀어) 및 AiShell2 (중국어)를 사용하였다.
- **모델 구성:**
  - Whisper: Teacher(Small, 244M) $\rightarrow$ Student(Tiny, 39M)
  - Wav2vec 2.0: Teacher(XLS-R 1B) $\rightarrow$ Student(XLS-R 300M)
- **평가 지표:** Character Error Rate (CER)를 사용하여 성능을 측정하였다.
- **비교 대상 (Baselines):** Normal KD, Super Loss, Focal Loss, Annealing KD.

### 주요 결과

- **정량적 성능:** 다수의 데이터셋에서 AdaKD가 가장 낮은 CER을 기록하며 우수한 성능을 보였다.
  - CV-Hindi: 23.27% (Normal KD 25.59%)
  - CV-Chinese: 25.20% (Normal KD 26.52%)
  - CV-Tamil: 12.07% (Normal KD 13.39%)
- **특이 사항:** AiShell2 데이터셋의 경우, 데이터 규모가 매우 크기 때문에 vanilla KD나 단순 Fine-tuning이 제안 방법보다 약간 더 좋은 성능을 보였다. 이는 데이터셋이 충분히 클 경우 증류 기법의 이점이 상대적으로 감소할 수 있음을 시사한다.
- **Ablation Study:** 임계값 $t$를 평균 외에 25, 50, 75 백분위수로 설정하여 실험한 결과, 데이터셋의 분포에 따라 최적의 $t$ 값이 달라질 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점

AdaKD는 샘플의 난이도를 교사 모델의 손실이라는 객관적인 지표로 수치화하고, 이를 통해 학습의 우선순위를 동적으로 조정함으로써 모델 압축 성능을 향상시켰다. 특히 특정 아키텍처에 종속되지 않고 손실 함수 상단에 추가하는 방식이므로 범용성이 높다는 점이 큰 강점이다.

### 한계 및 비판적 해석

1. **하이퍼파라미터 의존성:** $t$와 $k$라는 두 가지 하이퍼파라미터를 수동으로 튜닝해야 한다는 점이 한계로 지적된다. 특히 $k$의 범위가 모델(Whisper vs Wav2vec 2.0)에 따라 매우 크게 차이 난다는 점은 최적 값을 찾기 위한 실험 비용을 증가시킨다.
2. **대규모 데이터셋에서의 효용성:** AiShell2 결과에서 보듯, 데이터가 극도로 많을 때는 정교한 증류 전략보다 단순 학습이 더 효과적일 수 있다. 이는 본 기법이 특히 데이터가 부족하거나 효율적인 전이가 절실한 상황에서 더 큰 가치를 가짐을 의미한다.
3. **가정의 타당성:** 교사 모델의 손실이 높다고 해서 반드시 그 샘플이 '풍부한 지식'을 가졌다고 단정할 수 없다. 단순히 노이즈가 심하거나 잘못 레이블링 된(noisy label) 샘플일 가능성도 배제할 수 없는데, 이에 대한 필터링 기전이 부족하다.

## 📌 TL;DR

본 논문은 ASR 모델 압축을 위해 교사 모델의 손실을 기반으로 샘플 난이도를 측정하고, 이에 따라 증류 가중치를 동적으로 조절하는 **AdaKD** 기법을 제안한다. 커리큘럼 학습 원리를 적용해 쉬운 샘플에서 어려운 샘플 순으로 지식을 전이하며, Whisper와 wav2vec 2.0 모델 실험을 통해 기존 KD 및 instance-level 손실 함수들보다 낮은 CER을 달성하였다. 이 연구는 향후 하이퍼파라미터의 학습 가능(learnable) 구조 도입 및 Pruning/Quantization과의 결합을 통해 더 효율적인 모델 압축 프레임워크로 확장될 가능성이 크다.
