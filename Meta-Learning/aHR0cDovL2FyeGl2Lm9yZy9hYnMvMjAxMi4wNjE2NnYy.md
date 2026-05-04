# Few-Shot Segmentation Without Meta-Learning: A Good Transductive Inference Is All You Need?

Malik Boudiaf, Hoel Kervadec, Ziko Imtiaz Masud, Pablo Piantanida, Ismail Ben Ayed, Jose Dolz (2021)

## 🧩 Problem to Solve

본 논문은 Few-Shot Segmentation(FSS) 분야에서 지배적인 패러다임인 Meta-Learning(메타 학습)의 한계를 지적하고, 이를 대체할 수 있는 효율적인 추론 방법론을 제안한다.

기존의 FSS 연구들은 주로 Episodic Training(에피소드 기반 학습)을 통해 모델이 새로운 클래스에 빠르게 적응하도록 설계되었다. 그러나 이러한 접근 방식은 다음과 같은 세 가지 주요 문제를 가진다. 첫째, 테스트 단계에서의 Support Shot 수가 학습 단계에서의 설정과 다를 경우 성능이 빠르게 포화(Saturation)되는 경향이 있다. 둘째, 학습 데이터와 테스트 데이터의 도메인이 서로 다른 Domain Shift 상황에서 일반화 능력이 떨어진다. 셋째, 복잡한 메타 학습 구조로 인해 학습 과정이 복잡하며, 실제 환경에서의 유연성이 부족하다.

따라서 본 논문의 목표는 Meta-Learning 없이 표준적인 Cross-Entropy 학습만으로 특징 추출기(Feature Extractor)를 구축하고, 추론 단계에서 쿼리 이미지의 통계적 특성을 활용하는 Transductive Inference(전이적 추론)를 통해 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 추론 단계에서 쿼리 이미지의 라벨링되지 않은 픽셀들의 통계적 정보를 활용하여 분류기를 최적화하는 **RePRI(Region Proportion Regularized Inference)** 방법론을 도입하는 것이다. 

핵심 직관은 단순히 Support Set의 라벨 정보에만 의존(Inductive Inference)하는 것이 아니라, 쿼리 이미지 내의 전경(Foreground)과 배경(Background)의 비율(Proportion) 정보를 정규화 도구로 사용하여 과적합을 방지하고 예측의 정밀도를 높이는 데 있다. 이를 통해 메타 학습 없이도 기존 최신 모델(SOTA)과 경쟁하거나 이를 뛰어넘는 성능을 달성할 수 있음을 보여준다.

## 📎 Related Works

기존의 Few-Shot Learning 연구는 크게 Gradient-based 방법과 Metric-learning 기반 방법으로 나뉜다. 특히 FSS에서는 Prototypical Networks에서 영감을 받아 Support 이미지에서 클래스 프로토타입을 생성하고, 이를 쿼리 이미지와 비교하는 Two-branch 또는 Single-branch 아키텍처가 주를 이루어 왔다.

하지만 이러한 방식들은 대부분 Inductive Inference, 즉 학습된 가중치를 고정하고 입력된 Support Set에 기반해 예측하는 방식에 의존한다. 반면, 이미지 분류 분야에서는 테스트 시점에 unlabeled 데이터를 함께 고려하는 Transductive Inference가 성능 향상을 가져온다는 점이 입증되었다. 본 논문은 이러한 전이적 추론 개념을 세그멘테이션으로 확장하며, 특히 단순한 Entropy Minimization만으로는 세그멘테이션의 특성상 자명한 해(Trivial Solution)에 빠질 수 있다는 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인
본 방법론은 크게 두 단계로 나뉜다. 첫째, Base 클래스들에 대해 표준적인 Cross-Entropy 손실 함수를 사용하여 특징 추출기 $f_\phi$를 학습시킨다. 이때 복잡한 에피소드 학습을 배제한다. 둘째, 추론 단계에서 각 Task에 대해 단순한 선형 분류기를 정의하고, 쿼리 이미지의 특성을 반영한 특수 손실 함수를 통해 분류기 파라미터 $\theta$를 최적화한다.

### Transductive Objective (손실 함수)
추론 시 각 Task에 대해 다음과 같은 목적 함수를 최소화한다.

$$ \min_{\theta} CE + \lambda_H H + \lambda_{KL} D_{KL} $$

각 항의 의미는 다음과 같다.

1. **Cross-Entropy ($CE$):** Support 이미지의 라벨링된 픽셀들에 대해 계산되는 표준 교차 엔트로피이다. 이는 분류기가 Support Set의 정보를 학습하게 하지만, 1-shot 상황에서는 쿼리 이미지에 대해 과적합되어 매우 좁은 영역만 활성화되는 문제를 일으킨다.
2. **Shannon Entropy ($H$):** 쿼리 이미지 픽셀들에 대한 예측 결과의 엔트로피를 계산한다.
   $$ H = -\frac{1}{|\Psi|} \sum_{j \in \Psi} p_Q(j)^\top \log(p_Q(j)) $$
   이 항은 모델이 쿼리 이미지에 대해 더 확신 있는(Confident) 예측을 하도록 유도하며, 결정 경계를 데이터 밀도가 낮은 지역으로 밀어내는 역할을 한다.
3. **KL-Divergence Regularizer ($D_{KL}$):** 모델이 예측한 전경/배경 비율 $\hat{p}_Q$와 목표 비율 $\pi$ 사이의 거리를 좁히는 정규화 항이다.
   $$ D_{KL} = \hat{p}_Q^\top \log \left( \frac{\hat{p}_Q}{\pi} \right), \quad \text{where } \hat{p}_Q = \frac{1}{|\Psi|} \sum_{j \in \Psi} p_Q(j) $$
   이 항은 전경의 비율을 강제함으로써 $CE$와 $H$만으로 발생할 수 있는 퇴행적 해(Degenerate solutions)를 방지한다.

### 분류기 설계 및 최적화
분류기는 단순한 선형 분류기를 사용하며, 학습 가능한 파라미터 $\theta = \{w, b\}$ (프로토타입 벡터와 바이어스)로 구성된다. 픽셀 $j$의 예측 확률 $p(j)$는 다음과 같이 코사인 유사도를 기반으로 계산된다.

$$ s^{(t)}_\square(j) = \text{sigmoid}\left( \tau \left[ \cos(z_\square(j), w^{(t)}) - b^{(t)} \right] \right) $$
$$ p^{(t)}_\square(j) = \begin{pmatrix} 1 - s^{(t)}_\square(j) \\ s^{(t)}_\square(j) \end{pmatrix} $$

여기서 $\tau$는 온도 하이퍼파라미터이다. $w^{(0)}$는 Support 전경 특징들의 평균으로 초기화하고, $b^{(0)}$는 쿼리 이미지의 초기 전경 예측 평균으로 설정한다. 이후 SGD를 통해 $w$와 $b$를 최적화한다.

### 비율 파라미터 $\pi$의 결정
$\pi$에 대한 정보가 없을 경우, 모델의 예측 비율 $\hat{p}_Q$를 사용하여 $\pi$를 함께 학습한다. 구체적으로는 초기 단계에서는 $\hat{p}_Q^{(0)}$를 사용하다가, 특정 반복 횟수 $t_\pi$ 이후에 업데이트된 $\hat{p}_Q^{(t_\pi)}$로 $\pi$를 갱신하여 스스로 정규화하는 방식을 취한다.

## 📊 Results

### 실험 설정
- **데이터셋:** $\text{PASCAL-5}^i$ 및 $\text{COCO-20}^i$를 사용한다.
- **백본:** ResNet-50 및 ResNet-101 기반의 PSPNet을 사용한다.
- **지표:** mIoU(mean Intersection over Union)를 측정한다.
- **비교 대상:** RPMM, PFENet 등 최신 Meta-learning 기반 FSS 모델들과 비교한다.

### 주요 결과
1. **표준 벤치마크 성능:** 1-shot 설정에서는 기존 SOTA 모델들과 경쟁 가능한 수준의 성능을 보였으며, 5-shot 설정에서는 $\text{PASCAL-5}^i$ 기준 SOTA 대비 약 5~6%의 성능 향상을 보였다.
2. **Shot 수 증가에 따른 강건성:** 메타 학습 기반 모델들은 학습 시의 shot 수와 테스트 시의 shot 수가 다를 때 성능이 정체되는 경향이 있으나, RePRI는 shot 수가 증가할수록(1 $\to$ 5 $\to$ 10 shot) 성능 향상 폭이 더 커지는 양상을 보였다.
3. **Domain Shift 상황:** $\text{COCO-20}^i$로 학습하고 $\text{PASCAL-VOC}$로 테스트하는 교차 도메인 설정에서 RePRI가 기존 방법론들보다 훨씬 우수한 성능을 기록하여, 일반화 능력이 뛰어남을 입증하였다.
4. **Oracle 실험:** 쿼리 이미지의 정확한 전경 비율 $\pi^*$를 미리 알고 있을 때(Oracle case), mIoU가 비약적으로 상승(최대 14% 향상)하는 것을 확인하였다. 이는 전경 비율 정보가 매우 강력한 정규화 도구가 될 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 FSS에서 Meta-learning이 반드시 필수적인 요소가 아니며, 오히려 추론 단계에서의 정교한 Transductive Inference가 더 중요할 수 있음을 시사한다. 

특히 주목할 점은 **전경/배경 비율 정보의 중요성**이다. $CE$ 손실만으로는 Support Set에 과적합되어 쿼리 이미지의 작은 영역만 선택하는 문제가 발생하는데, 이를 $\pi$를 통한 KL-divergence 정규화로 해결하였다. Oracle 실험 결과는 정확한 크기 예측 방법만 개발된다면 추가적인 아키텍처 변경 없이도 성능을 획기적으로 높일 수 있다는 가능성을 보여준다.

다만, 추론 시마다 최적화 과정을 거쳐야 하므로 단순 Forward pass만 수행하는 기존 모델들보다 추론 속도(FPS)가 다소 느리다는 점이 한계로 지적된다. 하지만 그 차이가 수용 가능한 범위 내에 있으며, 성능 이득이 훨씬 크다는 점이 강조된다.

## 📌 TL;DR

본 논문은 복잡한 Meta-learning 없이 표준 Cross-Entropy 학습과 새로운 전이적 추론 기법인 **RePRI**를 통해 Few-Shot Segmentation 성능을 높였다. RePRI는 쿼리 이미지의 전경 비율 정보를 정규화에 활용하여 과적합을 방지하며, 특히 5-shot 이상의 시나리오와 도메인 변화가 있는 환경에서 기존 SOTA 모델들을 능가하는 강건함을 보여준다. 이 연구는 향후 FSS 연구가 복잡한 메타 학습 구조보다는 효율적인 추론 최적화와 객체 크기 추정으로 방향을 전환할 수 있는 중요한 근거를 제시한다.