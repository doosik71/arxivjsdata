# UNSUPERVISED SEMANTIC SEGMENTATION BY DISTILLING FEATURE CORRESPONDENCES

Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, William T. Freeman

## 🧩 Problem to Solve

- **레이블링 부담**: 시맨틱 분할(Semantic Segmentation)은 픽셀 단위로 객체를 분류하여 세밀한 이해를 가능하게 하지만, 고품질의 레이블링된 학습 데이터가 대규모로 필요하다는 큰 한계를 가집니다. 특히 의료, 생물학, 천체물리학 등 복잡한 도메인에서는 레이블을 정의하거나 제공하는 것이 어렵거나 불가능할 수 있습니다.
- **기존 비지도 학습의 비효율성**: 기존의 비지도 시맨틱 분할(Unsupervised Semantic Segmentation, USSS) 방법들은 의미론적으로 의미 있는 특징을 학습하는 것과, 학습된 특징들이 명확하고 응집력 있는 클러스터를 형성하도록 압축하는 과정을 하나의 종단 간(end-to-end) 프레임워크에서 동시에 수행하려 했지만, 그 성능에 제약이 있었습니다.

## ✨ Key Contributions

- **특징 상관관계의 의미론적 일관성 입증**: 최신 비지도 심층 신경망(예: DINO)에서 추출된 조밀한 특징(dense features) 간의 상관관계 패턴이 실제 시맨틱 레이블과 놀랍도록 일관된다는 것을 경험적으로 입증했습니다.
- **STEGO 프레임워크 제안**: 이러한 관찰을 바탕으로, 비지도 특징을 고품질의 이산(discrete) 시맨틱 레이블로 증류(distill)하는 새로운 트랜스포머 기반 아키텍처인 STEGO (Self-supervised Transformer with Energy-based Graph Optimization)를 제안합니다.
- **새로운 대비 손실 함수 개발**: 특징들이 이미지 말뭉치(corpora) 전반의 관계를 보존하면서도 응집력 있는 클러스터를 형성하도록 장려하는 새로운 대비(contrastive) 손실 함수를 핵심으로 사용합니다.
- **최첨단 성능 달성**: CocoStuff 데이터셋에서 +14 mIoU, Cityscapes 데이터셋에서 +9 mIoU를 달성하며 기존 최첨단 비지도 시맨틱 분할 방법 대비 크게 향상된 성능을 보여주었습니다.
- **아키텍처 설계 정당화**: CocoStuff 데이터셋에 대한 제거 연구(ablation study)를 통해 STEGO의 각 구성 요소 및 설계 선택의 효과를 분석하고 정당화했습니다.

## 📎 Related Works

- **자기지도 시각 특징 학습**: 노이즈 제거(denoising), 인페인팅(inpainting), 직소 퍼즐(jigsaw puzzles), 색상화(colorization), 회전 예측(rotation prediction) 등 다양한 대리 작업(surrogate task)을 통해 학습하는 방법들이 있었습니다. 최근에는 대비 학습(contrastive learning)이 주류를 이루며 (예: MoCoV2, SimCLR), 특히 DINO와 같이 spatially dense한 특징 맵을 생성하는 접근 방식이 주목받았습니다. STEGO는 이러한 사전 학습된 자기지도 특징을 입력으로 활용합니다.
- **비지도 시맨틱 분할**: IIC(Invariant Information Clustering), Contrastive Clustering, SCAN, PiCIE(IIC 개선) 등이 있으며, 주로 변환 불변성(transformation invariance)과 공변성(equivariance)을 활용하여 의미론적 특징을 학습하고 클러스터링을 수행합니다. SegSort는 슈퍼픽셀(superpixel)을, MaskContrast는 saliency mask를 활용합니다. STEGO는 기존 특징을 정제하여 클러스터 형성을 유도합니다.
- **Visual Transformers (ViT)**: 컨볼루션 신경망(CNN)의 한계를 극복하고 긴 범위 상호작용 모델링에 효과적인 트랜스포머는 자연어 처리(NLP)를 넘어 컴퓨터 비전에서도 널리 사용됩니다. 특히 DINO(Caron et al., 2021)는 self-distillation을 통해 자기지도 학습 프레임워크 내에서 ViT를 사용하여 시맨틱하게 의미 있는 객체 분할을 생성합니다. STEGO는 DINO의 임베딩 품질을 활용하지만, 다른 심층 네트워크 특징과도 호환됩니다.

## 🛠️ Methodology

STEGO는 비지도 방식으로 사전 학습된 시각적 백본(backbone)에서 추출된 특징의 상관관계를 새로운 손실 함수를 통해 정제하여 시맨틱 분할 결과를 도출합니다.

1. **특징 상관관계 분석 ($F_{hwij}$)**:
    - 두 이미지 $x, y$의 조밀한 특징 맵 $f \in \mathbb{R}^{C \times H \times W}$와 $g \in \mathbb{R}^{C \times I \times J}$를 사용합니다.
    - 이들의 코사인 유사도를 기반으로 특징 상관관계 텐서 $F_{hwij} := \sum_{c} \frac{f_{chw}}{|f_{hw}|} \frac{g_{cij}}{|g_{ij}|}$를 계산합니다. 이 텐서의 값은 두 픽셀 간의 의미론적 유사도를 나타냅니다.
    - 이 특징 상관관계가 실제 레이블 공동 발생(ground truth label co-occurrence)과 강력하게 상관됨을 정량적으로 보였습니다.

2. **특징 증류를 위한 분할 헤드 학습**:
    - 사전 학습된 백본 네트워크 $N$은 고정된 상태로 유지하고, 경량의 분할 헤드 $S: \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{K \times H \times W}$를 학습합니다 ($K < C$).
    - $S$는 입력 특징을 $K$차원 코드 공간 $s \in \mathbb{R}^{K \times H \times W}$로 비선형 투영하여 특징 상관관계 패턴을 증폭하고 응집력 있는 클러스터를 형성하도록 합니다.

3. **대비(Contrastive) 손실 함수 설계**:
    - **기본 상관관계 손실**: $L_{simple-corr}(x,y,b) := - \sum_{hwij} (F_{hwij} - b)S_{hwij}$
        - $S_{hwij}$는 분할 특징 $s, t$ 간의 코사인 유사도인 분할 상관관계 텐서입니다.
        - 하이퍼파라미터 $b$는 '음의 압력'을 추가하여 특징 붕괴(collapse)를 방지합니다.
    - **안정화 및 학습 신호 개선**:
        - **0-클램프(0-Clamp)**: $max(S_{hwij}, 0)$를 적용하여 약하게 상관된 분할 특징이 직교하도록 장려하며, 최적화 불안정성을 줄입니다.
        - **공간 중심화(Spatial Centering, SC)**: $F_{SC_{hwij}} := F_{hwij} - \frac{1}{IJ} \sum_{i'j'} F_{hwi'j'}$를 적용하여 특징 상관관계를 공간적으로 중심화함으로써 작은 객체에 대한 학습 신호를 개선하고 최적화를 균형 있게 만듭니다.
    - **최종 상관관계 손실**: $L_{corr}(x,y,b) := - \sum_{hwij} (F_{SC_{hwij}} - b)max(S_{hwij},0)$

4. **STEGO 아키텍처 및 학습**:
    - **총 손실 함수**: 이미지 $x$와 자기 자신($x,x$), $x$의 K-최근접 이웃(KNN) 이미지($x,x_{knn}$), 그리고 무작위 이미지($x,x_{rand}$) 간의 세 가지 손실 인스턴스화를 사용합니다:
        $$L = \lambda_{self} L_{corr}(x,x,b_{self}) + \lambda_{knn} L_{corr}(x,x_{knn},b_{knn}) + \lambda_{rand} L_{corr}(x,x_{rand},b_{rand})$$
        - $\lambda$ 값과 $b$ 값은 학습 신호의 균형을 조절합니다.
    - **5-Crop 학습**: 이미지의 세부 해상도와 KNN 품질을 높이기 위해 학습 전에 5-Crop 기법을 적용합니다.
    - **후처리**: 학습된 분할 특징에 코사인 거리를 기반으로 한 미니배치 K-평균을 적용하여 픽셀을 클러스터링하고, 최종적으로 조건부 랜덤 필드(Conditional Random Field, CRF)를 사용하여 공간 해상도를 추가 정제합니다.

## 📊 Results

- **CocoStuff (27개 클래스):**
  - STEGO는 비지도 mIoU에서 +14, 비지도 Accuracy에서 +6.9의 상당한 성능 향상을 달성하며 기존 최첨단 방법인 PiCIE를 크게 능가했습니다.
  - 선형 프로브(linear probe) 평가에서도 mIoU에서 +26, Accuracy에서 +21의 큰 폭의 개선을 보여, 학습된 특징의 우수한 품질을 입증했습니다.
  - DINO, MoCoV2, ImageNet 지도학습 ResNet50 백본의 특징을 단순히 클러스터링하는 것보다 우수하여, 특징 증류 학습의 효과를 입증했습니다.
- **Cityscapes (27개 클래스):**
  - STEGO는 비지도 mIoU에서 +8.7, 비지도 Accuracy에서 +7.7의 개선을 보이며 모든 기준선 대비 뛰어난 성능을 입증했습니다.
- **정성적 결과:**
  - STEGO는 PiCIE에 비해 이미지 내의 말의 다리, 개별 새와 같은 세밀한 디테일을 훨씬 더 잘 분할합니다.
  - Cityscapes 데이터셋에서도 사람, 도로, 보도, 자동차, 표지판 등을 높은 디테일과 충실도로 성공적으로 식별했습니다.
- **오류 분석:**
  - "음식(things)"과 "음식(stuff)" 카테고리, "천장"과 "벽", "실내", "액세서리", "원료", "섬유" 등과 같이 모호하거나 정의가 불분명한 클래스 간의 혼동 오류가 관찰되었습니다. 이는 비지도 분할 방법 평가의 본질적인 어려움을 시사합니다.

## 🧠 Insights & Discussion

- **자기지도 특징의 잠재력**: 최신 자기지도 학습 백본(예: DINO)이 생성하는 심층 특징 간의 상관관계는 실제 레이블 공동 발생(ground truth label co-occurrence)과 직접적으로 연관되어 있으며, 이는 강력하고 완전히 비지도적인 학습 신호로 활용될 수 있음을 보여줍니다.
- **특징 증류의 중요성**: 단순히 사전 학습된 특징을 클러스터링하는 것을 넘어, STEGO가 제안하는 특징 증류(feature distillation) 프로세스는 이러한 비지도 학습 신호를 증폭하고 데이터셋 전반에 걸쳐 일관성을 높여 최종 분할 성능을 크게 향상시킵니다.
- **이론적 기반**: STEGO의 손실 함수가 통계 물리학의 Potts 모델(또는 연속 Ising 모델) 에너지 함수 최소화와 등가임을 보임으로써, 에너지 기반 그래프 최적화 관점에서 프레임워크를 이론적으로 정당화하고, CRF 추론과의 연관성을 제시했습니다. 이는 STEGO의 강력한 이론적 기반을 제공합니다.
- **모델의 한계**:
  - 레이블 온톨로지가 자의적인(arbitrary) 경우 (예: "음식"의 Things/Stuff 분류, "벽"과 "천장"의 구분) STEGO는 혼동을 겪을 수 있습니다. 이러한 상황에서 선형 프로브는 특징 품질의 더 중요한 척도가 될 수 있습니다.
  - 아직 지도학습 시스템과의 성능 격차가 존재하며, 순수 비지도 벤치마크에서의 추가적인 발전 가능성이 남아 있습니다.
  - 하이퍼파라미터 튜닝이 접지(ground truth) 데이터에 대한 교차 검증 없이 수동으로 이루어져 향후 자동화될 필요가 있습니다.

## 📌 TL;DR

이 논문은 비지도 시맨틱 분할(USSS)의 한계를 해결하기 위해, 사전 학습된 자기지도 시각 백본(예: DINO)의 조밀한 특징 간 **의미론적으로 일관된 상관관계**를 활용하는 **STEGO (Self-supervised Transformer with Energy-based Graph Optimization)**를 제안합니다. STEGO는 **새로운 대비 손실 함수**를 통해 이 특징 상관관계를 고품질의 이산 시맨틱 레이블로 **증류(distill)**합니다. 이 손실 함수는 특징들이 데이터셋 전반의 관계를 유지하면서 응집력 있는 클러스터를 형성하도록 유도하며, Potts 모델의 에너지 최소화와 연결됩니다. 그 결과, STEGO는 CocoStuff (+14 mIoU) 및 Cityscapes (+9 mIoU) 데이터셋에서 **기존 최첨단 방법 대비 압도적인 성능 향상**을 달성하며, 비지도 시맨틱 분할 분야의 큰 진전을 보여줍니다.
