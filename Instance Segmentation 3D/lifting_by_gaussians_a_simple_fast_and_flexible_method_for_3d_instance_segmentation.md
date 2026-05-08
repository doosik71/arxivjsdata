# Lifting by Gaussians: A Simple, Fast and Flexible Method for 3D Instance Segmentation

Rohan Chacko, Nicolai Häni, Eldar Khaliullin, Douglas Lee, Lin Sun (2025)

## 🧩 Problem to Solve

본 논문은 최근 고품질의 신규 뷰 합성(Novel View Synthesis)을 위해 등장한 3D Gaussian Splatting (3DGS) 필드에서 **오픈 월드 인스턴스 분할(Open-world Instance Segmentation)**을 효율적으로 수행하는 문제를 해결하고자 한다.

기존의 3D 장면 분할 방식들은 다음과 같은 한계점을 가지고 있었다. 첫째, 3D 데이터의 부족으로 인해 2D 분할 데이터를 3D로 리프팅(lifting)하는 방식을 사용하는데, 이 과정에서 다각도 뷰 간의 일관성을 강제하기 위해 비용이 많이 드는 전처리가 필요하거나, 3D 재구성(reconstruction)과 시맨틱 학습이 얽혀 있어 학습 시간이 매우 길다는 단점이 있다. 둘째, 알파 블렌딩(alpha-blending) 기반의 학습 방식은 경계 영역에서 시맨틱 정보가 섞이는 'semantic bleeding' 현상을 야기하여 추출된 3D 자산(asset)의 품질을 저하시킨다.

따라서 본 논문의 목표는 별도의 씬(scene)별 추가 학습 없이, 기존의 3DGS 필드를 활용하여 객체(object), 부분(part), 하위 부분(subpart)으로 구성된 계층적 3D 인스턴스 분할을 빠르고 정확하게 수행하는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **학습 기반의 최적화 대신, 2D 파운데이션 모델의 마스크를 3DGS의 기하학적 구조에 직접 투영(Lifting)하고 이를 점진적으로 병합(Incremental Merging)하는 방식**이다.

구체적인 핵심 기여 사항은 다음과 같다.

1. **Training-free 3D Instance Segmentation**: 경사 하강법 기반의 최적화 없이, 2D-to-3D 리프팅 전략과 기하학적/시맨틱 유사도 기반의 병합 절차를 통해 매우 빠르게 3D 분할을 수행한다.
2. **Max-contributor Gaussian Assignment**: 픽셀에 기여하는 여러 가우시안 중 가장 기여도가 높은 단 하나의 가우시안에만 ID를 할당함으로써, 기존 방식의 고질적인 문제인 semantic bleeding을 억제하고 깨끗한 객체 경계를 확보한다.
3. **Hierarchical Decomposition**: SAM(Segment Anything Model)의 다양한 스케일(whole, part, subpart)을 계층적으로 적용하여, 장면을 객체 $\rightarrow$ 부분 $\rightarrow$ 하위 부분으로 정밀하게 분해한다.
4. **신규 평가 프로토콜 제안**: 단순히 2D 마스크 렌더링 성능(mIoU)을 측정하는 것을 넘어, 개별적으로 추출된 3D 자산의 시각적 품질을 정량적으로 평가하는 새로운 프로토콜을 도입하였다.

## 📎 Related Works

논문에서는 3D 장면 이해를 위한 기존 접근 방식을 크게 두 가지로 분류하여 설명한다.

1. **Feature Distillation (특징 증류)**: CLIP, DINO, SAM과 같은 2D 파운데이션 모델의 특징을 3D 표현체로 리프팅하는 방식이다. 주로 역렌더링(inverse rendering)이나 특징 집계(feature aggregation)를 사용한다. 하지만 가우시안마다 고차원 벡터를 학습시키는 것은 메모리 사용량이 많고 렌더링 속도를 저하시키며, 학습 시간이 길다는 한계가 있다.
2. **2D Mask Lifting (2D 마스크 리프팅)**: SAM과 같은 모델의 2D 마스크를 3D 공간으로 투영하는 방식이다. Gaussian Grouping과 같은 연구는 트래킹 알고리즘을 통해 뷰 간 일관성을 맞추려 했으나, 시점 변화가 클 때 오류가 발생한다. 또한, 대조 학습(contrastive learning) 기반의 방식들은 데이터 집약적이며 연산 비용이 높고, 재구성과 분할 학습이 결합되어 있어 유연성이 떨어진다.

LBG는 이러한 기존 방식들과 달리, **그라디언트 기반 학습을 완전히 제거**하고 3DGS의 명시적 구조를 활용하여 직접적으로 시맨틱 정보를 할당함으로써 효율성과 속도를 획기적으로 개선하였다.

## 🛠️ Methodology

LBG의 전체 파이프라인은 2D 마스크 추출, 3D 리프팅, 점진적 병합, 그리고 계층적 분해의 단계로 구성된다.

### 1. 2D Mask and Feature Extraction

먼저 포즈 정보가 주어진 RGB 이미지 시퀀스 $\{I_1, I_2, \dots, I_t\}$에 대해 SAM을 사용하여 클래스 불가지론적(class-agnostic) 2D 분할 마스크 $\{m_j^t\}$를 추출한다. 이때 SAM의 특성을 이용해 `whole`, `part`, `subpart` 세 가지 수준의 마스크를 얻으며, 동시에 CLIP와 DINOv2를 사용하여 각 영역의 시맨틱 특징 벡터 $f_j^t$를 추출한다.

### 2. 2D-to-3D Lifting (Max-contributor Assignment)

기존의 방식들은 임계값을 기준으로 여러 가우시안에 ID를 할당하여 경계가 뭉개지는 현상이 발생했다. LBG는 각 픽셀 $p \in m_j^t$에 대해, 렌더링 시 가장 큰 가중치를 가진 단 하나의 가우시안 $i^*$만을 선택하여 ID를 부여한다.

$$i^* = \arg \max_i \left( \alpha'_i \prod_{j=1}^{i-1} (1 - \alpha'_j) \right)$$

여기서 $\alpha'_i$는 해당 가우시안이 픽셀에 기여하는 불투명도 값이다. 이 방식을 통해 3D 공간 상의 객체 파편(object fragments) $o_j^t = \langle G_j^t, f_j^t \rangle$을 생성한다.

### 3. Incremental Merging

새로운 프레임에서 생성된 객체 파편들을 기존의 3D 시맨틱 맵 $O_{t-1}$과 병합한다. 병합 여부는 다음 두 가지 지표를 통해 결정된다.

- **기하학적 중첩도 (Geometric Overlap)**: $\phi^{geom}(i, j) = \frac{|G_i^t \cap G_j^{t-1}|}{|G_i^t|}$ (두 파편 간 공유되는 가우시안의 비율)
- **시맨틱 유사도 (Semantic Similarity)**: $\phi^{sem}(f_i^t, f_j^{t-1}) = \frac{f_i^t \cdot f_j^{t-1}}{2}$ (특징 벡터 간의 정규화된 코사인 유사도)

LBG는 이 유사도 점수가 가장 높은 기존 객체와 새로운 파편을 탐욕적(greedily)으로 병합하며, 일치하는 객체가 없을 경우 새로운 객체를 생성한다. 병합 후 시맨틱 특징은 이동 평균(running average) 방식으로 업데이트된다.

### 4. Hierarchical Decomposition & Post-processing

먼저 `whole` 수준의 마스크로 전체 구조를 잡은 뒤, 동일한 리프팅 및 병합 과정을 `part`와 `subpart` 수준에 순차적으로 적용하여 계층적 씬 그래프를 구축한다.
마지막으로, 통계적 아웃라이어 제거(statistical outlier removal)와 3D 연결 성분 분석(connected component analysis)을 통한 분할 및 병합(split-and-merge) 과정을 거쳐 최종적으로 정제된 3D 자산을 추출한다.

## 📊 Results

### 실험 설정

- **데이터셋**: LERF (실제 환경 아이폰 캡처), 3D-OVS (롱테일 객체 카테고리)
- **비교 대상**: Gaussian Grouping, SAGA
- **지표**:
  - 2D 마스크 렌더링: mIoU
  - 3D 자산 추출 품질: PSNR, SSIM, LPIPS (추출된 객체를 50개 각도에서 렌더링하여 GT와 비교)

### 주요 결과

1. **처리 속도**: 표 1에 따르면 LBG의 전체 처리 시간은 약 450초로, Gaussian Grouping(3,922초)이나 SAGA(5,206초)보다 **약 10배 이상 빠르다**. 특히 3D 분할 단계에서는 단 27초만이 소요된다.
2. **3D 자산 품질**: 표 2에서 LBG는 모든 씬(Figurines, Ramen, Teatime)에서 SAGA와 Gaussian Grouping을 압도하는 PSNR 및 SSIM 성능을 보였다. 이는 max-contributor 할당 방식이 semantic bleeding을 효과적으로 억제했음을 증명한다.
3. **2D 마스크 합성**: 표 3에서 LBG는 학습 기반 최적화를 수행하지 않았음에도 불구하고, SAGA와 대등하거나 Gaussian Grouping보다 우수한 mIoU 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

LBG의 가장 큰 강점은 **"단순함에서 오는 효율성"**이다. 복잡한 최적화 루프를 제거하고 3DGS의 명시적인 렌더링 가중치를 활용함으로써 연산 비용을 획기적으로 줄였다. 특히 Mini-Splatting 표현체를 사용하여 가우시안의 수를 10배 줄이고 뷰 일관성 점수(view consistency score)를 도입해 플로터(floater, 공중에 떠다니는 노이즈)를 제거한 것이 3D 자산의 깨끗한 경계를 만드는 데 결정적인 역할을 하였다.

### 한계 및 논의사항

- **모델 로딩 시간**: 전체 처리 시간은 짧으나, 파운데이션 모델(SAM, CLIP 등)을 로드하는 시간이 상당하여 실시간 애플리케이션 적용에는 여전히 제약이 있다.
- **소형 객체 분할**: 현재의 병합 방식으로는 매우 작은 객체를 캡처하는 데 어려움이 있을 수 있다. 저자들은 이를 해결하기 위해 최소한의 학습 반복을 통한 미세 조정(fine-tuning) 단계를 제안한다.
- **모델 의존성**: Fast-SAM과 같은 경량화 모델을 사용할 경우 처리 속도는 4배 빨라지지만, 분할 품질(mIoU)이 급격히 저하되는 현상이 관찰되었다. 이는 3D 리프팅의 품질이 입력 2D 마스크의 정밀도에 강하게 의존함을 시사한다.

## 📌 TL;DR

본 논문은 학습 없이 2D 파운데이션 모델의 마스크를 3DGS 필드에 직접 투영하여 인스턴스를 분할하는 **Lifting by Gaussians (LBG)** 방법을 제안한다. 픽셀당 최대 기여 가우시안(max-contributor)을 선택하는 단순한 전략만으로 기존 학습 기반 방법론보다 **10배 빠른 속도**와 **더 깨끗한 3D 객체 추출 성능**을 달성하였다. 이 연구는 AR/VR에서의 객체 조작이나 대규모 3D 장면 이해를 위한 실용적인 파이프라인을 제공하며, 향후 3D 자산 생성 및 편집 분야에 중요한 기초 기술이 될 가능성이 높다.
