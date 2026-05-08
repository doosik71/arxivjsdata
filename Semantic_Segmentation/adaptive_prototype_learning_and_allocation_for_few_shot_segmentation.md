# Adaptive Prototype Learning and Allocation for Few-Shot Segmentation

Gen Li, Varun Jampani, Laura Sevilla-Lara, Deqing Sun, Jonghyun Kim, Joongkyu Kim

## 🧩 Problem to Solve

기존 Few-shot Segmentation (FSS) 방법론은 대부분 서포트 이미지에서 단일 프로토타입을 추출하여 쿼리 이미지의 객체 정보를 나타냅니다. 그러나 이 단일 프로토타입 방식은 모든 객체 정보를 포괄하기에 모호하며, 객체의 규모, 형태 변화, 부분적인 가려짐(occlusion)에 취약합니다. 또한, 픽셀 단위의 특징 매칭은 공간 정보를 손실하거나 과적합되기 쉽습니다. 본 연구는 이러한 한계를 극복하고, 다양한 객체 변화에 유연하게 대처할 수 있는 적응형 프로토타입 학습 및 할당 전략을 제안합니다.

## ✨ Key Contributions

* **Adaptive Superpixel-guided Network (ASGNet) 제안:** 다양한 객체 스케일, 형태, 가려짐에 적응 가능한 유연한 프로토타입 학습 기반 Few-shot Segmentation 접근 방식을 제시합니다.
* **두 가지 새로운 모듈 도입:**
  * **Superpixel-guided Clustering (SGC):** 서포트 이미지의 특징 공간에서 유사한 특징 벡터들을 클러스터링하여 다수의 대표적인 프로토타입을 생성합니다. 이는 파라미터가 없고(parameter-free) 별도의 학습 과정이 필요 없습니다(training-free).
  * **Guided Prototype Allocation (GPA):** SGC로 생성된 프로토타입 중 각 쿼리 픽셀에 가장 적합한 프로토타입을 선택적으로 할당하여 더 정확하고 적응적인 가이드를 제공합니다. 이들은 효과적인 플러그 앤 플레이(plug-and-play) 컴포넌트로 활용될 수 있습니다.
* **뛰어난 성능과 효율성:** 적은 수의 파라미터와 낮은 연산량으로 Pascal-5$^{\text{i}}$ 및 COCO-20$^{\text{i}}$ 데이터셋에서 Few-shot Segmentation의 최신 기술(SOTA)을 뛰어넘는 성능을 달성했습니다. 특히, 5-shot 세분화에서 COCO-20$^{\text{i}}$ 대비 SOTA를 5% 이상 능가하는 결과를 보였습니다.

## 📎 Related Works

* **Semantic Segmentation:** 주로 FCN(Fully Convolutional Network) 기반으로 다중 스케일 특징 집계 및 어텐션 메커니즘을 사용하지만, 대규모 픽셀 단위 라벨링 데이터가 필요하며 학습 데이터에 없는 새로운 클래스에는 일반화하기 어렵습니다.
* **Few-shot Learning:**
  * **Metric Learning:** 이미지 또는 영역 간의 거리/유사성 측정에 중점을 둡니다 (e.g., Prototypical Networks).
  * **Meta Learning:** 빠른 학습 능력을 위한 특정 최적화 또는 손실 함수를 정의합니다.
* **Few-shot Segmentation:** Shaban et al. [24]이 초기 연구를 시작했으며, PL [4], SG-One [43], PANet [33], CANet [41] 등은 프로토타입 기반 접근 방식을, PGNet [40], BriNet [37], DAN [32] 등은 픽셀-투-픽셀 연결 기반의 어피니티 학습(affinity learning) 방식을 제안했습니다. PMMs [35]는 EM 알고리즘으로 다중 프로토타입을 생성했지만, 모든 프로토타입이 동일한 중요도를 가집니다.
* **Superpixel Segmentation:** 유사한 특성(색상, 텍스처 등)을 가진 픽셀 그룹으로, Few-shot Segmentation의 기본 단위로 사용됩니다 [18, 21]. 본 연구는 maskSLIC [12]과 SSN [13]의 아이디어에서 영감을 받았습니다.

## 🛠️ Methodology

ASGNet(Adaptive Superpixel-guided Network)은 두 가지 핵심 모듈인 SGC(Superpixel-guided Clustering)와 GPA(Guided Prototype Allocation)를 통합합니다.

1. **공유 CNN 특징 추출:**
    * 서포트 이미지($I_{\text{S}}$)와 쿼리 이미지($I_{\text{Q}}$)는 ImageNet으로 사전 학습된 공유 CNN(백본: ResNet)을 통해 특징 맵($F_{\text{S}}, F_{\text{Q}}$)을 추출합니다.

2. **Superpixel-guided Clustering (SGC) 기반 프로토타입 추출:**
    * **좌표 및 마스크 적용:** 서포트 특징 맵 $F_{\text{S}}$에 픽셀 좌표를 연결하고, 서포트 마스크 $M_{\text{S}}$를 사용하여 객체 영역 내의 특징($F'_{\text{S}}$)만 추출합니다.
    * **반복 클러스터링:** 추출된 특징들을 대상으로 반복적인 클러스터링을 수행합니다. 각 반복에서, 각 픽셀과 슈퍼픽셀 중심 간의 연관성 $Q^t_{pi}$는 $e^{-\|F'_p - S^{t-1}_i\|^2}$와 같은 특징 공간에서의 유사성을 기반으로 계산됩니다. 새로운 슈퍼픽셀 중심 $S^t_i$는 마스크된 특징의 가중 합으로 업데이트됩니다.
    * **적응형 프로토타입 수:** 객체 스케일에 따라 슈퍼픽셀 중심의 개수($N_{\text{sp}}$)를 $N_{\text{sp}} = \min(\lfloor N_{\text{m}}/S_{\text{sp}} \rfloor, N_{\text{max}})$와 같이 적응적으로 조절합니다. ($N_{\text{m}}$은 마스크된 픽셀 수, $S_{\text{sp}}$는 각 초기 슈퍼픽셀 시드에 할당된 평균 영역, $N_{\text{max}}$는 최대 프로토타입 수).

3. **Guided Prototype Allocation (GPA) 기반 적응형 매칭:**
    * **유사성 측정:** 각 프로토타입 $S_i$와 쿼리 특징 맵의 각 픽셀 $F^{x,y}_{\text{Q}}$ 간의 코사인 유사성 $C^{x,y}_i = \frac{S_i \cdot F^{x,y}_{\text{Q}}}{\|S_i\| \cdot \|F^{x,y}_{\text{Q}}\|}$을 계산합니다.
    * **가이드 맵 및 특징 생성:** 각 쿼리 픽셀 위치에서 가장 유사한 프로토타입의 인덱스를 $G_{x,y} = \arg \max_i C^{x,y}_i$로 결정하여 가이드 맵($G$)을 생성하고, 이를 통해 픽셀별 가이드 특징($F_{\text{G}}$)을 만듭니다. 또한, 모든 슈퍼픽셀에 대한 유사성 정보를 합산하여 확률 맵($P$)을 얻습니다.
    * **쿼리 특징 정제:** 원래 쿼리 특징 $F_{\text{Q}}$, 가이드 특징 $F_{\text{G}}$, 확률 맵 $P$를 채널 차원으로 연결($F'_{\text{Q}} = f(F_{\text{Q}} \oplus F_{\text{G}} \oplus P)$)하고 $1 \times 1$ 컨볼루션을 적용하여 정제된 쿼리 특징 $F'_{\text{Q}}$를 생성합니다.

4. **다중 스케일 특징 집계 및 K-shot 설정 확장:**
    * 정제된 쿼리 특징은 Feature Enrichment Module과 FPN(Feature Pyramid Network)과 유사한 구조를 통해 다중 스케일 정보를 통합합니다.
    * K-shot 설정의 경우, 각 K개의 서포트 이미지-마스크 쌍에서 SGC로 얻은 모든 슈퍼픽셀 중심 $S = (S_1, S_2, \dots, S_k)$을 GPA에 통합하여 더 넓은 선택지를 제공하고 추가적인 연산 비용 없이 성능을 향상시킵니다.

## 📊 Results

* **Pascal-5$^{\text{i}}$ 데이터셋:**
  * ResNet-101 백본 사용 시, 5-shot 세분화에서 mIoU 64.36%를 달성하여 기존 SOTA(PFENet) 대비 2.40% 높은 성능을 보였습니다. 1-shot에서는 SOTA와 동등한 수준을 유지했습니다.
  * FB-IoU(Foreground-Background IoU) 기준으로는 5-shot에서 75.2%를 달성하여 기존 SOTA(PFENet 73.9%)를 능가했습니다. 또한, 1-shot 결과 대비 5-shot에서 가장 큰 성능 향상(RN-50 기준 5.0%)을 보였습니다.
  * 다른 SOTA 방법론에 비해 적은 수의 학습 가능한 파라미터(10.4M)를 가집니다.
* **COCO-20$^{\text{i}}$ 데이터셋:**
  * ResNet-50 백본 사용 시, 1-shot에서 mIoU 34.56%, 5-shot에서 mIoU 42.48%를 달성하여 두 설정 모두에서 SOTA를 기록했습니다.
  * RPMMs [35] 대비 1-shot에서 3.98%, 5-shot에서 6.96%의 상당한 mIoU 향상을 보였습니다.
* **정성적 결과:** 서포트 및 쿼리 이미지 간의 외관, 포즈, 스케일 및 가려짐 변화가 큰 경우에도 ASGNet이 정확한 세분화 결과를 생성하는 능력을 입증했습니다.

## 🧠 Insights & Discussion

* **다중 프로토타입의 유효성:** 기존 단일 프로토타입의 한계를 넘어, SGC를 통해 객체의 다양한 부분을 대표하는 다수의 프로토타입을 생성함으로써 복잡한 객체의 정보를 더욱 풍부하고 명확하게 표현할 수 있음을 입증했습니다.
* **적응형 할당의 중요성:** GPA는 각 쿼리 픽셀에 가장 관련성 높은 프로토타입을 동적으로 할당함으로써 쿼리 이미지의 내용(객체 형태, 가려짐 등)에 유연하게 대응하여 정확도를 높였습니다. 이는 특히 가려진 객체 부분에 대한 모델의 강건성을 크게 향상시킵니다.
* **효율적인 K-shot 확장:** 기존 K-shot 방법론들이 추가적인 연산 비용을 수반하는 복잡한 특징 병합 전략을 사용하는 반면, ASGNet은 각 샷에서 추출된 프로토타입들을 단순히 통합함으로써 추가 연산 없이도 상당한 성능 향상을 달성했습니다. 이는 GPA 모듈이 더 넓은 프로토타입 풀에서 최적의 선택을 함으로써 이점을 얻는다는 것을 시사합니다.
* **한계 및 개선점:** 1-shot 설정에서 SGC만 사용하여 프로토타입을 추출할 경우, 때로는 성능이 저하될 수 있는데, 이는 단일 서포트 샘플에서 너무 많은 프로토타입이 생성되어 서로 유사해지면 코사인 거리가 이를 효과적으로 구별하지 못할 수 있기 때문입니다. 그러나 GPA와 결합될 때 이러한 문제는 효과적으로 극복됩니다.

## 📌 TL;DR

Few-shot Segmentation에서 단일 프로토타입의 한계를 해결하기 위해, 본 논문은 **Adaptive Superpixel-guided Network (ASGNet)**를 제안합니다. ASGNet은 **Superpixel-guided Clustering (SGC)** 모듈로 서포트 이미지의 특징에서 적응적으로 다수의 프로토타입을 추출하고, **Guided Prototype Allocation (GPA)** 모듈로 각 쿼리 픽셀에 가장 관련성 높은 프로토타입을 동적으로 할당합니다. 이 적응형 다중 프로토타입 학습 및 할당 전략은 객체 스케일 및 형태 변화, 가려짐에 강건하며, 효율적인 K-shot 확장을 가능하게 합니다. 결과적으로 ASGNet은 Pascal-5$^{\text{i}}$ 및 COCO-20$^{\text{i}}$ 데이터셋에서 기존 최신 기술을 뛰어넘는 우수한 세분화 성능을 달성했습니다.
