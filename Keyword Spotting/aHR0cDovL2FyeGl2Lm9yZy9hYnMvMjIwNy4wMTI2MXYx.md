# Minimizing Sequential Confusion Error in Speech Command Recognition

Zhanheng Yang, Hang Lv, Xiong Wang, Ao Zhang, Lei Xie (2022)

## 🧩 Problem to Solve

본 논문은 리소스가 제한된 엣지 디바이스(Edge Device)에서 동작하는 음성 명령 인식(Speech Command Recognition, SCR) 시스템의 고질적인 문제인 '명령어 간 혼동(Command Confusion)' 문제를 해결하고자 한다.

일반적으로 엣지 디바이스에 탑재되는 모델은 크기가 작아 모델 용량이 제한적이며, 이로 인해 발음이 유사한 명령어들 사이에서 오인식하는 경우가 빈번하게 발생한다. 예를 들어, "에어컨을 켜줘"와 "에어컨을 꺼줘"와 같이 발음이 유사한 명령어는 실제 서비스 적용 시 완전히 상반된 동작을 수행하게 되므로, 이러한 혼동 오류는 사용자 경험에 치명적인 영향을 미친다. 따라서 본 연구의 목표는 명령어 간의 변별력을 높여 혼동 오류를 줄이는 새로운 훈련 기준을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 음성 인식의 판별적 훈련(Discriminative Training) 기법에서 영감을 얻어, SCR에 특화된 **MSCE(Minimize Sequential Confusion Error)** 훈련 기준을 제안한 것이다.

중심 아이디어는 단순히 타겟 명령어의 사후 확률을 높이는 것이 아니라, 타겟 명령어와 혼동하기 쉬운 다른 명령어들 사이의 변별력을 직접적으로 최적화하는 것이다. 이를 위해 Connectionist Temporal Classification(CTC) 손실 함수를 시퀀스 수준의 측정 함수로 도입하여 각 명령어의 우도(Likelihood)를 정의하고, 효율적인 학습을 위해 전체 비타겟(non-target) 집합 대신 전략적으로 선택된 '혼동 시퀀스 집합(Confusing Sequence Set)'을 구성하여 학습 리소스를 절약하면서도 혼동 오류를 효과적으로 줄이는 방법을 제안하였다.

## 📎 Related Works

기존의 SCR 접근 방식은 크게 두 가지로 나뉜다. 첫째는 HMM(Hidden Markov Model) 기반 방식으로, GMM이나 DNN을 음향 모델로 사용하고 Viterbi 디코더를 통해 최적의 경로를 찾는 방식이다. 둘째는 최근의 End-to-End(E2E) 신경망 모델 방식으로, CNN 등을 사용하여 특징에서 바로 명령어로 매핑한다. E2E 방식은 계산 비용이 낮고 구축이 간단하지만, 명령어의 길이가 다양해질수록 컨텍스트 모델링이 어렵고 새로운 명령어 확장 시 유연성이 떨어진다는 한계가 있다.

또한, 일반적인 훈련 기준인 Cross-Entropy(CE)나 CTC는 타겟 프레임 또는 시퀀스의 사후 확률을 최적화하는 데 집중할 뿐, 서로 다른 명령어 간의 판별 능력을 직접적으로 최적화하지 않는다. 이를 보완하기 위해 ASR(Automatic Speech Recognition) 분야에서는 LF-MMI, MWER, MPE 등 시퀀스 판별적 훈련 기준이 사용되어 왔으며, 특히 MCE(Minimum Classification Error)는 확률 분포 대신 분류 오류 자체를 직접 최적화하는 특성을 가진다. 본 논문은 이러한 MCE의 개념을 SCR의 명령어 혼동 문제에 맞게 변형하여 적용하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 아키텍처

시스템은 입력 음성에서 Fbank 특징을 추출하고, 이를 다중 스케일 확장 TDNN(Time-Delay Neural Network) 레이어 블록에 통과시켜 CD-phone(Context-Dependent phone)의 사후 확률 벡터를 출력하는 구조를 가진다.

* **Acoustic Model:** 모델의 크기와 수용 영역(Receptive Field)의 균형을 맞추기 위해 다양한 확장률(Dilation rate = 1, 2, 4)을 가진 TDNN 레이어를 배치하였다. 또한, 저지연(Low latency) 구현을 위해 모델 중간의 일부 블록이 왼쪽 컨텍스트만 참조하도록 설계하였다.
* **Decoding:** CD-phone을 모델링 단위로 사용하며, 결정 트리(Decision Tree)를 통해 유사한 HMM 상태를 묶어 모델 크기를 줄였다. 디코딩은 $\text{HCLG} = \min(\text{det}(\text{H} \circ \text{C} \circ \text{L} \circ \text{G}))$ 그래프 위에서 Viterbi 디코딩을 수행하며, 메모리 절약을 위해 래티스(Lattice) 생성 없이 고정 길이 배열과 두 개의 토큰 리스트만을 유지하는 최적화된 방식을 사용한다.

### 2. MSCE (Minimize Sequential Confusion Error) 기준

MSCE는 MCE의 판별 함수 개념을 시퀀스 수준으로 확장한 것이다.

**가. 측정 함수 (Measure Function)**
본 연구에서는 CTC 손실 함수를 시퀀스 수준의 측정 함수로 사용한다. 타겟 라벨 시퀀스 $l$에 대한 CTC 우도는 다음과 같이 정의된다.
$$L^{CTC}(l|x^T, \Lambda) = \sum_{\pi \in B^{-1}(l)} p(\pi|x^T, \Lambda)$$
여기서 $\pi$는 타겟 시퀀스의 모든 가능한 정렬 경로를 의미한다.

**나. 오분류 측정치 (Misclassification Measure)**
타겟 명령어 $\kappa$와 혼동 집합 $S_\psi$에 속한 혼동 명령어 $\psi$들 사이의 변별력을 극대화하기 위해 다음과 같은 측정치를 정의한다.
$$d^\kappa(x^T, \Lambda) = \frac{g^\kappa}{\sum_{\psi \in S_\psi} g^\psi}$$
여기서 분자는 타겟 명령어의 시퀀스 수준 우도를, 분모는 혼동 명령어들의 우도 합을 나타낸다. 이를 CTC 함수에 적용하면 최종적인 MSCE 기준식은 다음과 같다.
$$d^\kappa(x^T, \Lambda) = \frac{L^{CTC}(\kappa)}{\sum_{\psi \in S_\psi} L^{CTC}(\psi)}$$

### 3. 혼동 명령어 선택 전략 (Confusing Command Selection)

모든 비타겟 명령어를 계산하는 것은 GPU 메모리 부담이 크므로, 다음과 같은 세 가지 전략으로 혼동 집합 $S_\psi$를 구성한다.

* **Pronunciation Similarity Strategy (PSS):** 사전(Lexicon)을 통해 명령어를 음소(Phone) 수준으로 확장한 후, Levenshtein 거리를 계산하여 가장 유사한 상위 $N$개의 명령어를 선택한다.
* **Random Selection Strategy (RSS):** 전체 명령어 집합에서 무작위로 $N$개를 선택한다.
* **Hybrid Strategy (HS):** 유사 명령어 풀과 전체 명령어 풀을 모두 준비하고, 각각에서 일정 비율($i$개와 $N-i$개)로 무작위 추출하여 동적으로 구성한다.

### 4. 학습 절차 (Multi-Task Learning)

학습 속도를 높이고 과적합을 방지하기 위해 단계적 학습을 수행한다. 먼저 CE 기준으로 네트워크를 수렴시킨 후, MSCE와 CE를 함께 사용하는 멀티태스크 학습을 진행한다. 최종 손실 함수는 다음과 같다.
$$L = \beta L^{MSCE} + (1-\beta) L^{CE}$$
일반적으로 $\beta = 0.8$로 설정하여 사용한다.

## 📊 Results

### 1. 실험 환경 및 데이터셋

* **데이터셋:** 에어컨 제어를 위한 41개의 중국어 명령어(길이 2-6자)로 구성된 자체 수집 데이터셋을 사용하였다.
* **구성:** 학습 세트는 1,744명의 화자로부터 수집된 약 95k 구절이며, `pyroomacoustics`를 이용해 RIR(Room Impulse Response) 및 노이즈를 추가하여 950k까지 증강하였다. 또한 400k의 비명령어(Negative samples)를 추가하였다.
* **평가 지표:** ROC 곡선, $0.01$ FAR(False Alarm Rate)에서의 FRR(False Reject Rate), 그리고 혼동 오류(Confusion Error) 감소율을 측정하였다.

### 2. 주요 결과

* **훈련 기준 비교:** MSCE(HS 전략 적용)는 CE 베이스라인 대비 $0.01$ FAR에서 FRR을 상대적으로 **33.7% 감소**시켰다. 또한 LF-MMI와 비교했을 때도 FRR을 **12.8% 더 낮추는** 우수한 성능을 보였다.
* **혼동 오류 감소:** Table 1에 따르면, MSCE(HS)는 타 기준 대비 혼동 오류를 상대적으로 **18.28% 감소**시켜 가장 뛰어난 성능을 기록하였다.
* **선택 전략 비교:** HS $\approx$ RSS $>$ PSS 순으로 성능이 나타났다. 이는 단순한 발음 유사성(PSS)보다는 전체 클래스를 적절히 커버하는 샘플링(RSS, HS)이 더 중요하며, 여기에 유사 명령어-전략을 소량 추가한 HS가 가장 효과적임을 시사한다.

## 🧠 Insights & Discussion

본 논문은 모델의 파라미터 수나 추론 시간을 늘리지 않고도 훈련 기준의 변경만으로 명령어 혼동 문제를 획기적으로 줄일 수 있음을 입증하였다.

특히 흥미로운 점은 단독 CTC 훈련이 때로는 성능을 저하시킬 수 있다는 발견이다. CTC는 피크 사후 확률(Peak posterior)을 생성하는 경향이 있어, 프루닝 임계값(Pruning threshold)이 엄격할 경우 정답 토큰을 너무 빨리 제거(Prune)하여 거부 오류(Reject error)를 유발할 수 있다. 반면 MSCE는 타겟과 비타겟 간의 상대적 우도를 최적화함으로써 이러한 문제를 완화하고 변별력을 높인다.

또한, 혼동 집합 구성 시 발음이 유사한 것들만 모으는 것(PSS)보다 무작위 샘플링(RSS)이 더 효과적이었다는 점은, 모델이 특정 유사군뿐만 아니라 전반적인 클래스 경계를 명확히 학습하는 것이 더 중요하다는 통찰을 제공한다. 다만, 실제 환경에서의 다양한 소음 조건이나 더 방대한 명령어 집합에서의 확장성에 대해서는 추가적인 검증이 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 엣지 디바이스의 음성 명령 인식 시스템에서 발생하는 명령어 간 혼동 문제를 해결하기 위해, CTC 우도를 이용한 판별적 훈련 기준인 **MSCE(Minimize Sequential Confusion Error)**를 제안하였다. 제안 방법은 타겟 명령어와 혼동하기 쉬운 명령어 집합 사이의 변별력을 직접 최적화하며, 특히 유사도와 무작위성을 결합한 하이브리드 전략(HS)을 통해 **혼동 오류를 18.28% 감소**시키고 **FRR을 33.7% 개선**하였다. 이 연구는 추가적인 연산 비용 없이 훈련 단계의 최적화만으로 SCR 시스템의 정확도를 높일 수 있는 실용적인 방법을 제시하였다.
