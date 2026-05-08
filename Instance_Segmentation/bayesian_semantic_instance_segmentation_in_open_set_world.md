# Bayesian Semantic Instance Segmentation in Open Set World

Trung Pham, Vijay Kumar B G, Thanh-Toan Do, Gustavo Carneiro, and Ian Reid (2018)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Open-set 환경에서의 Semantic Instance Segmentation이다. 기존의 딥러닝 기반 인스턴스 분할 방법론들은 학습 과정에서 정의된 특정 클래스(Known classes)에 대해서만 작동하며, 학습 데이터셋에 포함되지 않은 미지의 객체(Unknown objects)가 등장할 경우 이를 탐지하거나 분할하지 못하는 한계가 있다.

이러한 문제는 특히 자율 주행 시스템이나 로봇 공학 분야에서 치명적이다. 실제 환경에서는 예측 불가능한 다양한 객체들이 등장하며, 시스템이 장면을 전체적으로 이해(Holistic scene understanding)하기 위해서는 알려진 객체뿐만 아니라 알려지지 않은 객체까지 모두 분할하여 추론할 수 있어야 하기 때문이다. 또한, 기존의 지도 학습 기반 방법론들은 모든 객체 인스턴스에 대한 정밀한 세그멘테이션 마스크(Annotation masks)를 요구하는데, 이는 비용이 매우 많이 들며 클래스 수가 무한히 증가하는 실제 시나리오에서는 사실상 불가능한 작업이다. 따라서 본 논문은 알려진 클래스에 대한 Bounding box 수준의 약한 지도 학습(Weakly supervision)만으로도 알려진 객체와 미지의 객체를 모두 분할할 수 있는 프레임워크를 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Bayesian 프레임워크를 활용하여 알려진 객체와 미지의 객체를 동시에 처리할 수 있는 글로벌 이미지 분할 접근 방식을 제안한 것이다.

가장 중심적인 설계 아이디어는 개별 탐지 결과에 대해 독립적으로 마스크를 생성하는 기존의 'Detect-and-segment' 방식에서 벗어나, 이미지 전체를 겹치지 않는 여러 영역으로 분할하는 글로벌 최적화 문제를 푸는 것이다. 이를 위해 알려진 객체는 Object detector의 결과(Bounding box 또는 Mask)를 통해 설명하고, 알려지지 않은 객체는 이미지의 경계선 정보(Boundary information)를 통한 지각적 그룹화(Perceptual grouping)를 통해 설명하도록 설계하였다. 또한, 방대한 이미지 분할 공간(Partition space)에서 최적해를 효율적으로 찾기 위해 Boundary-driven region hierarchy 기반의 효율적인 이미지 분할 샘플러와 Simulated Annealing 최적화 기법을 도입하였다.

## 📎 Related Works

기존의 인스턴스 분할 연구는 크게 두 가지 방향으로 나뉜다.

첫째, Mask R-CNN과 같은 지도 학습 기반 방법론들이다. 이들은 먼저 객체를 탐지한 후 각 탐지 영역에 대해 마스크를 생성하는 방식을 취한다. 그러나 이러한 방법들은 훈련 시 사용된 클래스 외의 객체는 완전히 무시하며, 정밀한 픽셀 단위의 마스크 주석이 필수적이라는 비용적 한계가 있다.

둘째, 비지도 학습 기반 분할 방법론들이다. 이들은 강한 지도 신호 없이도 미지의 객체를 발견할 수 있으나, 주로 색상, 텍스처, 에지(Edge)와 같은 저수준(Low-level) 시각 정보에 의존한다. 이로 인해 분할의 정확도가 상대적으로 낮으며 복잡한 장면에서 견고함이 떨어진다는 단점이 있다.

제안된 방법론은 이 두 가지 접근 방식의 장점을 결합한다. 알려진 객체에 대해서는 Detector의 정보를 활용해 정확도를 높이고, 동시에 비지도 학습의 일반성을 유지하여 미지의 객체까지 함께 분할함으로써 기존 방법론들의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문의 접근 방식은 입력 이미지 $I$를 서로 겹치지 않는 $k$개의 영역 $R_i$로 분할하고, 각 영역에 세만틱 라벨 $l_R$을 할당하는 문제로 정의한다. 전체 구조는 다음과 같은 조건이 충족되는 파티션을 찾는 것이다.
$$\bigcup_{i=1}^{k} R_i = \Omega, \quad R_i \cap R_j = \emptyset; \forall i \neq j$$

### Bayesian Formulation

본 논문은 세그멘테이션 솔루션 $S$를 $S = \{(R_1, t_1, \theta_1), \dots, (R_k, t_k, \theta_k)\}$로 정의하며, 여기서 $t_i$는 모델 타입, $\theta_i$는 해당 모델의 파라미터이다. 최적의 $S$는 다음과 같은 사후 확률(Posterior distribution)을 최대화하는 값으로 결정된다.
$$p(S|I) \propto p(I|S)p(S)$$

#### 1. Likelihood Models $p(I|S)$

이미지 영역을 설명하기 위해 세 가지 모델을 사용한다.

- **Boundary/Contour Model (C):** 미지의 영역을 설명한다. COB 네트워크를 통해 추정된 경계 확률 맵을 사용하며, 외부 경계는 강하고 내부 경계는 약한 영역이 객체일 가능성이 높다고 판단한다.
- **Bounding Box Model (B):** 알려진 객체를 설명한다. 탐지된 Bounding box $b$와 영역 $R$ 사이의 $\text{IoU}(b^R, b)$ 및 픽셀들의 공간적 분포(가우시안 분포)를 통해 가능도를 계산한다.
- **Mask Model (M):** 선택 사항으로, 탐지된 마스크 $m$과 영역 $R$ 사이의 $\text{IoU}(R, m)$을 사용하여 가능도를 계산한다.

#### 2. Prior Model $p(S)$

분할 결과의 물리적 타당성을 부여하기 위해 다음과 같은 Prior를 정의한다.
$$p(S) \propto \exp(-\lambda k) \cdot \prod_{i=1}^{k} \exp(-|R_i|^{0.9}) \cdot \exp(-\zeta(R_i))$$

- $\exp(-\lambda k)$: 영역의 개수 $k$가 너무 많아지는 것을 방지한다.
- $\exp(-|R_i|^{0.9})$: 너무 작은 영역보다 큰 영역이 생성되도록 유도한다.
- $\zeta(R_i)$: 영역의 픽셀 수와 해당 영역의 Convex hull 면적의 비율을 계산하여, 영역이 기하학적으로 조밀(Compact)하도록 강제한다.

### MAP Inference 및 Simulated Annealing

최적의 $S^*$를 찾기 위해 에너지 함수 $E(S, I) = -\log(p(S|I))$를 최소화하는 Simulated Annealing(SA) 최적화를 수행한다.

**Efficient Partition Sampling:**
탐색 공간이 매우 방대하므로, COB 네트워크 기반의 Region hierarchy를 구축하여 효율적으로 샘플링한다.

- **Sample-and-paste:** 하이러키 트리에서 무작위로 영역 $R$을 선택해 현재 파티션에 '붙여넣기' 함으로써, 영역의 분할(Split), 병합(Merge), 또는 분할 후 병합(Split-and-merge)을 확률적으로 수행한다.
- **Occlusion Handling:** 가려짐(Occlusion)으로 인해 분리된 영역들을 처리하기 위해, 동일한 Bounding box나 Mask 내에 존재하는 영역 쌍을 샘플링하여 병합 후보로 추가한다.

## 📊 Results

### 실험 설정

- **데이터셋:** NYU RGB-D (알려진 클래스 60개, 미지의 클래스 721개) 및 MS COCO 데이터셋을 사용하였다.
- **비교 대상:** 비지도 분할 후 $\text{IoU}$ 기반으로 라벨링하는 단순 Baseline 방법론, 그리고 완전 지도 학습 기반의 Mask R-CNN과 비교하였다.
- **평가 지표:** NYU 데이터셋에서는 $\text{IoU}$ 임계값(0.5, 0.75)에 따른 F-1 score를 측정하였고, COCO 데이터셋에서는 mAP 및 mIoU를 측정하였다.

### 주요 결과

1. **Open-set 성능 (NYU 데이터셋):**
   - 아무런 가이드가 없는 경우에도 제안 방법이 Baseline보다 알려진/미지의 클래스 모두에서 높은 F-1 score를 기록하였다.
   - Bounding box나 Mask 정보를 추가했을 때, 알려진 클래스에 대한 정확도가 크게 향상되었다. 반면 Baseline은 마스크 정보를 사용할 때 오히려 성능이 하락하는 경향을 보였다.
2. **약한 지도 학습 성능 (COCO 데이터셋):**
   - 제안 방법은 Bounding box 수준의 지도 학습(Weakly supervision)만으로도, 모든 인스턴스 마스크를 학습한 Mask R-CNN의 성능에 근접하는 경쟁력 있는 mIoU 결과를 보여주었다.
   - 특히, 탐지기가 놓친 객체(Miss-detected)나 미지의 객체(Unknown)까지도 성공적으로 분할해내는 정성적 우수성을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 인스턴스 분할을 '개별 객체의 마스크 생성' 문제가 아닌 '이미지 전체의 최적 파티션' 문제로 재정의함으로써 Open-set 환경의 문제를 효과적으로 해결하였다. 특히, 딥러닝 기반의 경계선 추출(COB)과 전통적인 Bayesian 최적화(Simulated Annealing)를 결합하여, 모델의 유연성(미지 객체 처리)과 정확성(알려진 객체 처리)을 동시에 확보한 점이 돋보인다.

**한계 및 논의사항:**

- **연산 효율성:** Simulated Annealing은 수천 번의 반복 계산이 필요하므로, 실시간성(Real-time)이 요구되는 시스템에 적용하기에는 추론 속도가 느릴 수 있다.
- **최적화 의존성:** 결과가 초기 하이러키 설정 및 SA의 하이퍼파라미터(온도 스케줄 등)에 영향을 받을 가능성이 있다.
- **비판적 해석:** 본 방법론은 end-to-end 학습 방식이 아니므로, 딥러닝 모델의 성능 향상이 곧바로 전체 시스템의 성능 향상으로 이어지기 위해서는 Bayesian 최적화 단계와의 정밀한 통합이 필요하다. 하지만, 사람이 일일이 마스크를 그리지 않고도 미지 객체를 식별하고 이를 점진적으로 학습 데이터에 추가할 수 있는 'Incremental annotation strategy'를 가능하게 했다는 점에서 실용적 가치가 매우 높다.

## 📌 TL;DR

본 논문은 알려진 객체뿐만 아니라 미지의 객체까지 모두 분할할 수 있는 **Bayesian Open-set Semantic Instance Segmentation** 프레임워크를 제안한다. Bounding box 정보와 이미지 경계선 정보를 결합한 Bayesian 가능도 모델을 구축하고, 이를 Simulated Annealing과 효율적인 Region hierarchy 샘플러로 최적화하여, 정밀한 마스크 주석 없이도 지도 학습 기반 방법론에 근접하는 성능을 달성하였다. 이 연구는 특히 미지의 객체를 식별해야 하는 자율 로봇 시스템의 장면 이해 능력을 향상시키고, 효율적인 데이터 증강 및 점진적 학습 체계를 구축하는 데 중요한 기초를 제공한다.
