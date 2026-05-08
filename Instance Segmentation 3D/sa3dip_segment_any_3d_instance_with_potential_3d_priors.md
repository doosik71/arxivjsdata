# SA3DIP: Segment Any 3D Instance with Potential 3D Priors

Xi Yang, Xu Gu, Xingyilang Yin, Xinbo Gao (2024)

## 🧩 Problem to Solve

본 논문은 개방형 환경(open-world)에서의 3D 인스턴스 분할(instance segmentation) 문제를 해결하고자 한다. 최근 연구들은 Segment Anything Model(SAM)과 같은 2D 파운데이션 모델의 제로샷(zero-shot) 능력을 3D 공간으로 확장하여 뛰어난 성과를 거두고 있다. 일반적으로 이러한 방식은 3D 장면을 기하학적 기본 단위인 superpoints로 분할한 뒤, SAM이 생성한 2D 마스크를 가이드로 삼아 이들을 병합하는 파이프라인을 따른다.

그러나 기존 방식들은 3D Prior(사전 정보)의 활용이 제한적이라는 치명적인 한계가 있다. 첫째, superpoints를 생성할 때 오직 공간 좌표 기반의 법선 벡터(normal)만을 고려하기 때문에, 벽과 칠판처럼 기하학적 구조가 유사한 서로 다른 인스턴스들을 하나의 그룹으로 묶어버리는 under-segmentation 문제가 발생한다. 둘째, SAM의 본질적인 특성인 '부분 단위 분할(part-level segmentation)' 경향이 3D 공간으로 전이되어, 하나의 객체가 여러 개의 작은 조각으로 나뉘는 over-segmentation 문제가 발생한다. 따라서 본 논문의 목표는 잠재적인 3D Prior를 적극적으로 활용하여 이러한 과소 분할 및 과잉 분할 문제를 해결하고 고품질의 3D 인스턴스 분할을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 2D 파운데이션 모델에 지나치게 의존하는 대신, 3D 데이터가 본래 가지고 있는 기하학적 및 텍스처 정보를 보완적으로 활용하여 분할의 정밀도를 높이는 것이다.

1. **보완적 프리미티브 생성(Complementary Primitives Generation):** 기하학적 정보(normals)뿐만 아니라 색상 정보(texture)를 함께 사용하여 superpoints를 생성함으로써, 기하학적으로는 유사하지만 색상이 다른 인스턴스들을 초기에 명확히 구분한다.
2. **3D Prior 기반의 인스턴스 정제(Instance-aware Refinement):** 3D Detector를 도입하여 3D 공간에서의 제약 조건을 제공함으로써, SAM의 영향으로 인해 과하게 분할된 인스턴스들을 다시 통합하는 과정을 추가한다.
3. **ScanNetV2-INS 데이터셋 제안:** 기존 ScanNetV2의 불완전한 어노테이션(미라벨링 및 누락된 인스턴스)을 수정하고 보완한 ScanNetV2-INS를 구축하여, 더 공정하고 정밀한 성능 평가 지표를 제공한다.

## 📎 Related Works

### 기존 연구 및 한계

- **Closed-set 3D Segmentation:** Mask3D와 같은 지도 학습 기반 방법들은 높은 성능을 보이지만, 대규모의 정밀한 3D 어노테이션 데이터가 필요하며 학습되지 않은 새로운 객체가 등장하는 open-world 시나리오에서는 적용이 어렵다.
- **Open-set 3D Segmentation:** SAM3D, SAMPro3D, SAI3D 등은 SAM의 2D 마스크를 3D로 투영하여 활용한다. 이들은 대개 3D 장면을 기하학적 프리미티브로 먼저 나누고 이를 병합하는 방식을 취하지만, 3D Prior의 활용 부족으로 인해 앞서 언급한 under-segmentation과 over-segmentation 문제에서 자유롭지 못하다.

### SA3DIP의 차별점

기존의 open-vocabulary 방법들이 주로 2D-3D 간의 정렬(alignment)이나 GNN을 통한 병합 알고리즘 설계에 집중했다면, SA3DIP는 **"3D 공간 자체의 고유한 특성(색상, 3D Bounding Box 등)"**을 파이프라인의 시작과 끝 단계에 배치하여 2D 모델의 한계를 물리적으로 보완한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조

전체 시스템은 크게 세 단계로 구성된다: (A) 기하 및 텍스처 기반의 보완적 프리미티브 생성 $\rightarrow$ (B) 2D 마스크 기반의 Scene Graph 구축 $\rightarrow$ (C) Region Growing 및 3D Detector 기반의 인스턴스 정제.

### 2. 상세 구성 요소 및 절차

#### A. 보완적 프리미티브 생성 (Complementary Primitives Generation)

기존의 기하학적 정보만으로는 부족하므로, 색상 정보를 추가하여 더 세밀한 superpoints를 생성한다. 두 점 $v_a, v_b$ 사이의 엣지 가중치 $w(v_a, v_b)$는 다음과 같이 계산된다:

$$w(v_a, v_b) = w^n \cdot \frac{n_a \cdot n_b}{\|n_a\|\|n_b\|} + w^c \cdot \sqrt{\sum_{k=1}^{3} (c_{ak} - c_{bk})^2}$$

여기서 $n$은 법선 벡터(normal), $c$는 색상(color)을 의미하며, $w^n$과 $w^c$는 각각 기하 및 텍스처 정보의 가중치이다. 이를 통해 기하적으로는 평면이지만 색상이 다른 경계면을 효과적으로 분리한다.

#### B. Scene Graph 구축 (Scene Graph Construction)

생성된 3D 프리미티브 $\{U_i\}$를 노드로, 이들 사이의 유사도를 엣지로 하는 그래프를 구축한다.

1. **2D 투영:** 핀홀 카메라 모델을 사용하여 3D 프리미티브를 2D 이미지 평면으로 투영한다.
2. **Affinity 계산:** SAM이 생성한 2D 마스크 $S^m$과 투영된 프리미티브 $U_i^m$ 사이의 매칭을 통해 히스토그램 벡터 $e_{i,m}$을 생성하고, 두 프리미티브 간의 코사인 유사도를 통해 프레임별 affinity $A_{i,j}^m$을 구한다.
3. **가중 합산:** 가시성(visibility) $V$를 가중치로 사용하여 최종 affinity $A_{i,j}$를 산출한다:
    $$A_{i,j} = \frac{\sum_{m=1}^{M} (\gamma_{i,j}^m A_{i,j}^m)}{\sum_{m=1}^{M} \gamma_{i,j}^m}, \quad \text{where } \gamma_{i,j}^m = V_i^m \cdot V_j^m$$

#### C. Region Growing 및 인스턴스 정제 (Refinement)

먼저 affinity $A_{i,j}$와 유클리드 거리 $D_{i,j}$를 결합한 신뢰도 점수 $\delta_{i,j}$를 사용하여 1차 병합을 수행한다:
$$\delta_{i,j} = \frac{1}{D_{i,j}} \cdot A_{i,j}$$

이후, SAM의 part-level 분할로 인한 over-segmentation을 해결하기 위해 **3D Detector**를 통한 정제 과정을 거친다 (Algorithm 1):

- 모든 Bounding Box $bb_i$를 부피 기준 오름차순으로 정렬한다.
- 박스 내부 점들 중 특정 인스턴스 ID $O'_i$가 차지하는 비율 $\sigma_i$가 임계값 $\delta_2$보다 크면, 해당 영역을 하나의 인스턴스로 재라벨링한다.
- 부피가 작은 객체부터 처리함으로써, 큰 객체 위에 놓인 작은 객체가 잘못 통합되는 것을 방지한다.

## 📊 Results

### 실험 설정

- **데이터셋:** ScanNetV2, ScanNetV2-INS(제안), ScanNet++, Matterport3D, Replica.
- **지표:** $mAP, AP_{50}, AP_{25}$ (Class-agnostic 기준).
- **구현:** 단일 RTX 4090 GPU 사용.

### 주요 결과

- **성능 향상:** ScanNetV2에서 $mAP$ 기준 기존 최고 수준 방법론들보다 유의미한 성능 향상을 보였으며, 특히 ScanNetV2에서 $mAP$가 41.6%를 기록하여 SAMPro3D(33.7%)나 SAI3D(30.8%)를 크게 상회한다.
- **강건성 검증:** 더 정밀한 라벨링이 적용된 ScanNetV2-INS에서도 경쟁력 있는 성능을 유지하며, 이는 본 모델이 실제로 더 세밀한 객체 인식 능력을 갖췄음을 시사한다.
- **범용성:** Matterport3D와 Replica 데이터셋에서도 다른 open-vocabulary 방법론 대비 높은 AP를 기록하여 데이터셋에 상관없이 강건한 성능을 보임을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 2D 파운데이션 모델의 강력한 성능을 활용하면서도, 그 한계(part-level segmentation, 3D 기하 정보 무시)를 **3D Prior**라는 물리적 제약 조건으로 보완했다는 점에서 매우 실용적인 접근을 취했다. 특히, 단순히 알고리즘을 개선한 것이 아니라 데이터셋의 결함(ScanNetV2의 라벨 누락)을 발견하고 이를 수정한 ScanNetV2-INS를 제시함으로써 학술적 기여도를 높였다.

### 한계 및 비판적 해석

1. **텍스처 정보의 불안정성:** 실험 결과 $w^n=0.96, w^c=0.04$와 같이 기하 정보에 훨씬 높은 가중치를 두었다. 이는 조명, 그림자, 반사 등으로 인해 색상 정보(RGB)가 단독으로는 신뢰하기 어렵다는 점을 보여준다. 즉, 텍스처 Prior는 보조적인 수단일 뿐 결정적인 해결책은 아님을 의미한다.
2. **계산 효율성과 정밀도의 트레이드-오프:** 효율성을 위해 3D Prior만으로 superpoints를 생성하는데, 고해상도 데이터나 조명 변화가 극심한 환경에서는 여전히 superpoints의 개수가 너무 많아지거나 경계가 불분명할 가능성이 있다.
3. **2D 모델 의존성:** 최종 병합의 핵심인 Affinity Matrix가 여전히 SAM의 결과에 의존하므로, SAM 자체가 완전히 틀린 마스크를 생성할 경우 이를 3D Prior만으로 완전히 복구하기에는 한계가 있을 수 있다.

## 📌 TL;DR

SA3DIP는 SAM과 같은 2D 파운데이션 모델을 이용한 3D 인스턴스 분할 시 발생하는 **under-segmentation(기하 유사성 문제)**과 **over-segmentation(부분 분할 문제)**을 해결하기 위해, **색상 기반의 보완적 프리미티브 생성**과 **3D Detector 기반의 인스턴스 정제** 과정을 도입한 프레임워크이다. 또한, 더 정확한 평가를 위해 개선된 **ScanNetV2-INS** 데이터셋을 함께 제안하였다. 이 연구는 2D 모델을 3D로 확장할 때 3D 고유의 Prior를 결합하는 것이 성능 향상의 핵심임을 입증하였으며, 향후 open-world 3D 이해 연구에 중요한 가이드라인을 제시한다.
