# The Devil is in the Boundary: Exploiting Boundary Representation for Basis-based Instance Segmentation

Myungchul Kim, Sanghyun Woo, Dahun Kim, In So Kweon (2020)

## 🧩 Problem to Solve

최근 실시간 비전 애플리케이션을 위해 단일 단계(single-stage) 인스턴스 분할(instance segmentation) 방식이 주목받고 있으며, 이는 기존의 2단계 방식보다 효율적인 설계와 우수한 전역 마스크 표현(global mask representation) 능력을 보여준다. 그러나 이러한 단일 단계 방식들은 인스턴스의 경계(boundary)를 정밀하게 묘사하는 능력이 여전히 부족하다는 문제점이 있다.

인스턴스의 경계 정보는 객체의 형태를 나타내는 강력한 표현 수단이 되지만, 기존의 전역 마스크 기반 방법들은 마스크 출력의 마지막 신호에 의존하여 암시적으로만 학습될 뿐, 경계 정보를 명시적으로 학습하지 않는다. 이로 인해 경계 부근의 픽셀 분류가 어려워지며, 결과적으로 마스크의 경계가 뭉툭하거나 불분명하게 생성되는 coarse mask 문제가 발생한다. 특히 객체가 서로 겹쳐 있거나 형태가 복잡한 경우 이러한 현상이 두드러진다. 본 논문의 목표는 전역 경계 표현을 통해 기존 전역 마스크 기반 방법의 부족한 고주파 세부 정보를 보완하고, 정밀한 인스턴스 경계를 추출하는 B2Inst(Boundary Basis based Instance Segmentation)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 경계 표현을 전역적 관점(Global view)과 인스턴스적 관점(Instance view) 모두에서 활용하여 분할 정밀도를 높이는 것이다.

1. **전역 이미지 경계 학습(Holistic Image Boundary Learning):** 개별 객체 단위의 RoI(Region of Interest) 경계가 아니라, 이미지 전체 수준에서 모든 인스턴스의 경계를 한 번에 학습하는 전역 경계 베이시스(global boundary basis)를 도입한다. 이를 통해 복잡한 장면이나 심한 가림(occlusion) 현상이 있는 상황에서도 객체 간의 경계를 더 명확히 구분할 수 있다.
2. **경계 인식 마스크 스코어링(Boundary-aware Mask Scoring):** 마스크의 면적뿐만 아니라 형태(경계)의 일치 여부를 함께 평가하는 통합 퀄리티 지표를 설계하였다. 이를 통해 추론 단계에서 경계 표현이 정확한 고품질의 마스크를 우선적으로 선택할 수 있게 한다.
3. **범용적 적용 가능성:** 제안된 방법론이 특정 구조에 국한되지 않고 YOLACT, BlendMask와 같은 기존의 다양한 베이시스 기반 단일 단계 프레임워크에 통합되어 일관된 성능 향상을 이끌어낼 수 있음을 증명하였다.

## 📎 Related Works

인스턴스 분할 연구는 크게 2단계(two-stage) 방식과 1단계(one-stage) 방식으로 나뉜다. Mask R-CNN으로 대표되는 2단계 방식은 바운딩 박스를 먼저 검출한 뒤 내부 마스크를 예측하는데, 이는 RoI-wise 특징 풀링에 의존하므로 추론 속도가 느리고 해상도가 낮은 마스크를 생성하는 경향이 있다.

반면, 최근의 1단계 방식 중 '베이시스 기반(basis-based)' 방법들(YOLACT, BlendMask 등)은 전역적인 베이시스 마스크 표현과 인스턴스별 계수를 결합하여 빠르게 마스크를 생성한다. 경계 학습과 관련하여 Boundary-preserving Mask R-CNN 같은 연구가 있었으나, 이는 2단계 방식의 RoI 내부 경계에만 집중하여 이미지 전체의 맥락을 파악하는 데 한계가 있었다. B2Inst는 이러한 한계를 극복하기 위해 전역적 관점의 경계 표현을 도입함으로써 기존 방식과 차별화된다.

## 🛠️ Methodology

B2Inst의 전체 아키텍처는 특징 추출을 위한 Backbone, 인스턴스 검출을 위한 Detection Head, 전역 베이시스를 생성하는 Global Basis Head, 그리고 마스크 품질을 평가하는 Mask Scoring Head의 네 부분으로 구성된다.

### 1. 전역 이미지 경계 학습 (Learning Holistic Image Boundary)

기존의 베이시스 헤드는 마스크 손실 함수에 의해서만 암시적으로 학습되었으나, B2Inst는 경계 정보를 명시적으로 학습시킨다.

- **경계 정답 생성:** 이진 마스크(binary mask) 정답 데이터에 라플라시안 연산자(Laplacian operator)를 적용하여 소프트 경계를 생성하고, 이를 임계값(threshold) 처리하여 최종 이진 경계 맵을 생성한다.
- **손실 함수:** 정밀한 경계 예측을 위해 Binary Cross-Entropy loss, Dice loss, 그리고 Boundary loss 세 가지를 함께 사용한다. 특히 Boundary loss는 후처리 없이도 테스트 단계에서 날카로운 경계를 예측하도록 돕는다.

### 2. 경계 인식 마스크 스코어링 (Boundary-aware Mask Scoring)

단순한 IoU(Intersection over Union)는 면적의 일치도만 측정하므로, 경계가 불분명함에도 IoU가 높게 나오는 문제가 있다. 이를 해결하기 위해 경계 일치도 $S_{boundary}$를 도입한다.

- **경계 스코어 정의:** 예측된 마스크 $M_{pred}$와 정답 마스크 $M_{gt}$에 라플라시안 커널 $\Delta f$를 적용하여 각각 경계 $B_{pred}$와 $B_{gt}$를 추출한다. 이후 두 경계 사이의 Dice metric을 계산하여 스코어를 산출한다.
$$S_{boundary} = \frac{2 \sum_{i}^{h \times w} B_{i}^{pred} B_{i}^{gt} + \epsilon}{\sum_{i}^{h \times w} (B_{i}^{pred})^2 + \sum_{i}^{h \times w} (B_{i}^{gt})^2 + \epsilon}$$
여기서 $\epsilon$은 0으로 나누는 것을 방지하기 위한 작은 값이다.

- **스코어링 헤드 구조:** 4개의 컨볼루션 레이어를 공유하며, 마지막에 $S_{IoU}$와 $S_{boundary}$를 각각 회귀(regression)하는 두 개의 FC(fully-connected) 브랜치로 구성된다. 입력으로는 예측 마스크 $M_{pred}$, 예측 경계 $B_{pred}$, 그리고 RoI-pooled FPN 특징 $F_{RoI}$를 결합하여 사용한다.

- **최종 추론 점수:** 추론 시에는 클래스 분류 점수 $S_{class}$와 예측된 $S_{IoU}$, $S_{boundary}$를 결합하여 최종 마스크 신뢰도 $S_{mask}$를 계산한다.
$$S_{mask} = S_{class} \cdot \sqrt{S_{IoU} \cdot S_{boundary}}$$

## 📊 Results

### 실험 설정

- **데이터셋:** MS COCO 2017 인스턴스 분할 벤치마크를 사용하였다.
- **평가 지표:** 표준 COCO-style AP 및 크기별 AP($AP_S, AP_M, AP_L$)를 측정하였다.
- **비교 대상:** Mask R-CNN, MS R-CNN, CondInst, BlendMask, PointRend 등 최신 1단계 및 2단계 모델들과 비교하였다.

### 주요 결과

- **요소별 기여도(Ablation Study):** 전역 경계 베이시스(HBB)를 추가했을 때 전체 AP가 0.4 상승했으며, 특히 $AP_{75}$와 $AP_S$(작은 객체)에서 큰 향상이 있었다. 경계 인식 스코어링(BS+MS)을 함께 적용했을 때 추가로 0.6 AP가 향상되어, 두 제안 요소가 상호 보완적으로 작동함을 확인하였다.
- **기존 프레임워크 결합:** B2Inst를 YOLACT와 BlendMask에 적용한 결과, 각각 2.0 AP와 1.0 AP의 성능 향상을 보였다. 이는 제안 방법이 베이시스 기반 구조에 범용적으로 적용 가능함을 시사한다.
- **SOTA 비교:** ResNet-50 및 ResNet-101 백본 기준, 동일 조건의 Mask R-CNN 및 최신 단일 단계 모델들(CenterMask, PointRend 등)보다 우수한 성능을 기록하였다. 특히 B2Inst-BlendMask (R-101)는 40.8 AP를 달성하며 경쟁력 있는 수치를 보였다.
- **정성적 결과:** 시각화 결과, 기존 BlendMask가 노이즈에 취약하거나 겹쳐진 객체의 경계를 뭉뚱그려 표현하는 것과 달리, B2Inst는 복잡한 형태와 가려진 부분의 경계를 훨씬 정밀하게 묘사하는 것으로 나타났다.

## 🧠 Insights & Discussion

본 연구는 단일 단계 인스턴스 분할에서 간과되었던 '경계 표현'의 중요성을 성공적으로 입증하였다.

**강점 및 통찰:**

1. **전역적 관점의 유효성:** 기존 2단계 방식의 RoI-wise 경계 학습과 달리, 이미지 전체의 경계를 학습함으로써 객체 간의 상호 관계와 가림 현상을 더 잘 해결할 수 있다는 점을 보여주었다.
2. **면적과 형태의 결합:** 마스크의 품질을 평가할 때 단순 면적(IoU)뿐만 아니라 형태(Boundary Dice)를 동시에 고려하는 것이 실제 시각적 품질과 더 높은 상관관계를 가지며, 이를 통해 더 정확한 마스크 선택이 가능함을 입증하였다.
3. **직교성(Orthogonality):** 제안된 방법이 특정 모델의 구조를 변경하는 것이 아니라 보조적인 학습 신호와 스코어링 메커니즘을 추가하는 방식이기에, 다양한 베이시스 기반 모델에 쉽게 이식될 수 있다는 점이 큰 장점이다.

**한계 및 논의:**
논문에서는 성능 향상을 입증했으나, 추가적인 경계 베이시스 채널과 스코어링 헤드로 인해 발생하는 연산 오버헤드와 추론 속도(FPS)의 소폭 감소에 대한 상세한 분석은 부족하다. 다만, 표 3에서 YOLACT와 BlendMask 적용 시 FPS 저하가 매우 미미함을 보여줌으로써 실시간성 저하가 크지 않음을 주장하고 있다.

## 📌 TL;DR

B2Inst는 단일 단계 인스턴스 분할 모델들이 겪는 '뭉툭한 경계' 문제를 해결하기 위해 **(1) 이미지 전체의 전역 경계 베이시스를 학습**하고, **(2) 면적과 경계 일치도를 동시에 고려하는 새로운 마스크 스코어링 방식**을 도입한 연구이다. 이 방법은 YOLACT, BlendMask 등 기존 모델에 쉽게 통합 가능하며, 특히 복잡한 형태나 겹쳐진 객체의 경계를 정밀하게 추출하여 COCO 데이터셋에서 SOTA 수준의 성능 향상을 달성하였다. 향후 실시간 정밀 분할 연구에 중요한 기여를 할 것으로 보인다.
