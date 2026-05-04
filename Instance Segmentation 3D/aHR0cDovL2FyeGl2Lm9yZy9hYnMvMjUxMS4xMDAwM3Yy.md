# DBGroup: Dual-Branch Point Grouping for Weakly Supervised 3D Semantic Instance Segmentation

Xuexun Liu, Xiaoxu Xu, Qiudan Zhang, Lin Ma, Xu Wang (2025)

## 🧩 Problem to Solve

본 논문은 3D 장면 이해의 핵심 과제인 3D 시맨틱 인스턴스 분할(3D Semantic Instance Segmentation)에서 발생하는 막대한 어노테이션 비용 문제를 해결하고자 한다. 기존의 완전 지도 학습(Fully Supervised) 방식은 모든 포인트에 대해 시맨틱 클래스와 인스턴스 ID를 지정해야 하므로 시간이 매우 많이 소요된다.

이를 완화하기 위해 제안된 기존의 약지도 학습(Weakly Supervised) 방식들, 즉 'One-Thing-One-Click(OTOC)' 방식이나 'Bounding Box(BBox)' 방식 역시 여전히 인스턴스를 개별적으로 구분해야 하는 노동 집약적인 과정이 필요하며, 복잡한 구조나 중첩된 객체를 처리하기 위해 전문가의 숙련도가 요구된다는 한계가 있다.

따라서 본 연구의 목표는 단순히 장면 내에 어떤 객체들이 존재하는지만 명시하는 **장면 수준 어노테이션(Scene-level annotation)**만을 사용하여, 어노테이션 비용을 획기적으로 낮추면서도 높은 성능의 3D 인스턴스 분할을 달성하는 DBGroup 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 2D-3D 교차 모달리티 정보를 활용하여 정교한 의사 라벨(Pseudo Label)을 생성하고, 이를 통해 모델을 점진적으로 학습시키는 두 단계(Two-stage) 파이프라인을 설계한 것이다.

1. **Dual-Branch Point Grouping**: 시맨틱 정보에 기반한 SGB(Semantic Guidance Branch)와 마스크 정보에 기반한 MGB(Mask Guidance Branch)를 통해 서로 다른 입도(Granularity)의 의사 라벨을 동시에 생성한다.
2. **의사 라벨 정제 전략**: SGB의 과소 분할(Under-segmentation)과 MGB의 과다 분할(Over-segmentation) 문제를 해결하기 위해 GAIM(Granularity-Aware Instance Merging)과 SSP(Semantic Selection and Propagation) 전략을 도입하였다.
3. **Instance Mask Filter (IMF)**: 독립적으로 생성된 시맨틱 라벨과 인스턴스 라벨 사이의 불일치를 해소하여 학습 노이즈를 줄이는 필터링 메커니즘을 제안하였다.
4. **효율적인 감독 체계**: 장면 수준 라벨만을 사용함에도 불구하고, 희소 포인트 수준(Sparse-point-level) 지도 학습 방식과 경쟁 가능한 성능을 달성하였으며, 기존의 장면 수준 시맨틱 분할 SOTA 성능을 넘어섰다.

## 📎 Related Works

기존의 약지도 3D 분할 연구는 주로 다음과 같은 접근 방식을 취했다.

- **Box-level annotation**: 인스턴스의 기하학적 정보를 제공하지만, 박스가 중첩되는 영역에서 포인트를 정확히 할당하는 것이 어렵다.
- **Sparse-points-level annotation**: Mean Teacher 구조나 그래프 기반 전파(Propagation) 방식을 사용하여 소수의 라벨을 확산시키지만, 여전히 인스턴스 단위의 포인트 선택 과정이 필요하다.
- **Scene-level annotation**: 주로 3D Class Activation Mapping(CAM)이나 2D-3D 정렬을 통해 시맨틱 분할에 집중해 왔으며, 인스턴스 분할 단계까지 효과적으로 확장한 연구는 부족했다.

DBGroup은 이러한 기존 방식들과 달리, 2D 사전 학습 모델(SAM, CLIP 등)의 강력한 일반화 능력을 3D 공간으로 투영하여 인스턴스 수준의 정보를 추출함으로써, 가장 낮은 수준의 감독(장면 수준 라벨)만으로도 인스턴스 분할을 수행한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

본 프레임워크는 **의사 라벨 생성 및 정제(Pseudo Label Generation and Refinement)** 단계와 **다회차 자기 학습(Multi-round Self-training)** 단계로 구성된다.

### 2. 의사 라벨 생성: Dual-Branch Point Grouping

- **Semantic Guidance Branch (SGB)**:
  - OpenSeg와 같은 사전 학습된 Vision-Language Model을 사용하여 다중 뷰 이미지에서 임베딩 $F^{2D}$를 추출하고, 이를 3D 포인트로 투영 및 평균화하여 $F^{3D}$를 얻는다.
  - 장면 수준 라벨의 텍스트 임베딩 $F^{1D}$와 $F^{3D}$ 간의 행렬 곱을 통해 클래스 스코어 $S$를 계산한다.
  - **BFS Grouping**: 반경 $r$ 내에서 동일한 시맨틱 예측을 가진 포인트들을 묶는 BFS 알고리즘을 통해 거친 입도의(Coarse-grained) 인스턴스 마스크 $\{M^q\}$를 생성한다.

- **Mask Guidance Branch (MGB)**:
  - 포인트 클라우드를 오버세그멘테이션하여 superpoint를 생성하고, 그 중심점을 SAM(Segment Anything Model)의 프롬프트로 사용하여 2D 마스크를 얻는다.
  - 2D 마스크들을 3D로 투영하고 다수결 투표(Voting) 방식을 통해 정밀한 입도의(Fine-grained) 인스턴스 마스크 $\{O^w\}$를 생성한다.

### 3. 의사 라벨 정제 (Pseudo Label Refinement)

- **Granularity-Aware Instance Merging (GAIM)**:
  - 거친 마스크 $M^q$와 정밀한 마스크 $O^w$ 사이의 중첩 비율 $\rho$를 계산한다.
  - $\rho$가 임계값 $\theta$보다 크면 $M^q$를 유지하고, 작으면 $M^q$가 여러 인스턴스를 포함한 것으로 간주하여 $O^w$와의 교집합 영역으로 분할하여 정제한다.
  - 이후 KNN을 이용해 너무 작은 인스턴스를 인접한 큰 인스턴스와 병합하여 최종 인스턴스 의사 라벨 $Y^I$를 확정한다.
- **Semantic Selection and Propagation (SSP)**:
  - 클래스별로 상위 $\alpha\%$의 고신뢰도 포인트만 선택하여 시맨틱 불균형을 방지한다.
  - 선택된 라벨을 superpoint 단위로 전파하여 공간적 연속성을 확보한 최종 시맨틱 의사 라벨 $Y^S$를 생성한다.

### 4. 자기 학습 네트워크 및 손실 함수

최종 네트워크는 3D U-Net 기반의 구조이며, 시맨틱 브랜치와 오프셋 브랜치로 나뉜다.

- **시맨틱 손실 ($L_{sem}$)**: 의사 라벨 $Y^S$를 사용하여 교차 엔트로피 손실을 적용한다.
  $$L_{sem} = -\frac{1}{N} \sum_{i=1}^N CE(\hat{Y}^S, Y^S)$$
- **오프셋 및 방향 손실 ($L_{off}, L_{dir}$)**: 포인트와 인스턴스 중심 간의 거리 차이를 $L_1$ 손실로 학습하고, 방향 벡터의 일치성을 위해 코사인 유사도 기반의 방향 손실을 추가한다.
  $$L_{off} = \frac{1}{N} \sum_{i=1}^N \|\hat{Y}^O - Y^O\|$$
  $$L_{dir} = -\frac{1}{N} \sum_{i=1}^N \frac{\hat{Y}^O}{\|\hat{Y}^O\|_{row2}} \cdot \frac{Y^O}{\|Y^O\|_{row2}}$$
- **Instance Mask Filter (IMF)**: 제안된 인스턴스 특징 $PF_i$에 대해 이진 마스크 $PM_i$를 예측하여 불필요한 포인트를 제거한다. 이때 $\text{BCE}$ 손실과 $\text{Dice}$ 손실을 함께 사용한다.
- **ScoreNet 손실 ($L_{sc}$)**: 최종 인스턴스 제안의 품질을 평가하는 ScoreNet을 학습시키며, $\text{IoU}$ 값에 따라 라벨을 부여하여 binary cross-entropy로 학습한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: ScanNetV2, S3DIS.
- **지표**: 시맨틱 분할은 mIoU, 인스턴스 분할은 mAP ($\text{AP}_{25}, \text{AP}_{50}, \text{AP}$)를 사용하였다.
- **구현 상세**: MinkowskiNet34C를 백본으로 사용하였으며, Adam 옵티마이저와 코사인 어닐링 스케줄러를 적용하였다.

### 2. 주요 결과

- **인스턴스 분할 성능**: ScanNetV2에서 DBGroup은 희소 포인트 수준 지도 학습 방식인 3D-WSIS와 대등한 성능을 보였으며, S3DIS에서는 3D-WSIS를 $\text{AP}$ 기준 3.8%, $\text{AP}_{50}$ 기준 7.5% 상회하는 결과를 보였다.
- **시맨틱 분할 성능**: 장면 수준 라벨을 사용한 기존 SOTA 방식들보다 ScanNetV2에서 7.2%, S3DIS에서 2.9% 높은 mIoU를 기록하며 월등한 성능을 입증하였다.
- **Precision vs Recall**: 포인트 수준 지도 방식보다 Precision은 낮았으나 Recall은 훨씬 높게 나타났다. 이는 장면 수준 라벨이 세밀한 위치 정보는 부족하지만, 객체의 전체적인 커버리지를 확보하는 데 유리하기 때문이다.

### 3. 절제 실험 (Ablation Study)

- **GAIM의 효과**: SGB 단독 또는 MGB 단독 사용보다 GAIM을 통해 두 입도를 결합했을 때 $\text{AP}$가 크게 상승하였으며, 이는 과소/과다 분할 문제를 동시에 해결했음을 보여준다.
- **SSP의 효과**: 클래스별 선택(Selection)과 superpoint 전파(Propagation)를 적용했을 때 mIoU가 단계적으로 상승하였다.
- **IMF의 효과**: IMF를 적용했을 때 $\text{AP}$가 25.4에서 26.5로 상승하여, 의사 라벨의 불일치로 인한 노이즈가 효과적으로 억제됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 3D 인스턴스 분할에서 가장 비용이 적게 드는 '장면 수준 라벨'만으로도 실용적인 수준의 성능을 낼 수 있음을 증명하였다. 특히, 2D 모델(SAM, VLM)의 결과물을 단순 투영하는 것에 그치지 않고, 입도 차이를 이용한 GAIM과 공간적 제약을 이용한 SSP라는 정제 과정을 거침으로써 의사 라벨의 품질을 높인 점이 인상적이다.

**강점**:

- 어노테이션 비용을 획기적으로 절감 (장면당 1분 미만).
- 2D-3D 교차 모달리티 정보를 효율적으로 융합하여 인스턴스 경계를 복원함.
- 자기 학습(Self-training) 루프를 통해 점진적으로 성능을 개선함.

**한계 및 논의**:

- **의존성**: 생성된 의사 라벨의 품질이 전적으로 사전 학습된 2D 모델(SAM, OpenSeg)의 성능에 의존한다. 만약 2D 모델이 객체를 오인식하면 3D 결과에도 직접적인 영향을 미친다.
- **계산 비용**: superpoint 생성 및 다중 뷰 이미지 처리에 따른 추가적인 계산 오버헤드가 발생한다.
- **정밀도 문제**: 결과에서 나타나듯 Precision이 낮은 경향이 있는데, 이는 장면 수준 라벨이 제공하는 정보의 희소성으로 인해 발생하는 근본적인 한계로 보이며, 이를 해결하기 위한 추가적인 제약 조건 연구가 필요할 것이다.

## 📌 TL;DR

본 논문은 장면 수준 라벨(Scene-level label)만을 사용하여 3D 시맨틱 인스턴스 분할을 수행하는 **DBGroup** 프레임워크를 제안한다. 이 모델은 시맨틱 가이드(SGB)와 마스크 가이드(MGB)의 듀얼 브랜치를 통해 서로 다른 입도의 의사 라벨을 생성하고, 이를 GAIM과 SSP 전략으로 정제하여 고품질의 지도 신호를 구축한다. 최종적으로 IMF 필터가 포함된 3D U-Net을 자기 학습 시킴으로써, 매우 적은 비용의 라벨링만으로도 기존의 희소 포인트 지도 학습 방식에 근접하는 성능을 달성하였다. 이 연구는 3D 데이터 어노테이션의 병목 현상을 해결할 수 있는 새로운 방향성을 제시한다.
