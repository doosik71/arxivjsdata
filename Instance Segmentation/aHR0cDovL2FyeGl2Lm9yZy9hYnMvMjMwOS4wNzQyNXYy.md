# JSMNet: Improving Indoor Point Cloud Semantic and Instance Segmentation through Self-Attention and Multiscale Fusion

Shuochen Xu, Zhenxin Zhang

## 🧩 Problem to Solve

본 논문은 실내 3D 포인트 클라우드(Indoor Point Cloud) 데이터의 의미론적 분할(Semantic Segmentation)과 인스턴스 분할(Instance Segmentation)을 동시에 수행하는 공동 분할(Joint Segmentation) 문제를 다룬다. 실내 공간의 포인트 클라우드 데이터는 서비스 로봇의 경로 계획, 내비게이션 시스템, 디지털 트윈 엔지니어링 등 다양한 응용 분야에서 필수적이다.

연구진이 해결하고자 하는 핵심 문제는 3D 포인트 클라우드 데이터 특유의 무질서함(disorder)과 비구조적 특성으로 인해 의미론적 특징을 효율적으로 추출하기 어렵다는 점이다. 특히, 고품질의 분할 결과를 얻기 위해서는 장거리 문맥 정보(long-range context information)를 제공하는 글로벌 특징(Global features)의 확보가 필수적이지만, 기존 방식으로는 이를 충분히 반영하기 어려웠다. 또한, 스캐너와 대상체 사이의 거리에 따라 포인트 클라우드의 밀도가 달라지는 특성으로 인해 발생하는 특징 손실 문제를 해결해야 할 필요성이 있었다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 전역적 수용장(Global receptive field)과 PointConv의 지역적 특징 추출 능력을 결합하고, 다양한 해상도의 특징을 적응적으로 융합하여 세밀한 표현력을 얻는 것이다. 주요 기여 사항은 다음과 같다.

1. **글로벌 특징 셀프 어텐션 인코딩 모듈 설계**: Transformer와 PointConv를 결합하여 포인트 클라우드의 강건한 특징 표현을 가능하게 하는 새로운 인코딩 레이어를 제안한다.
2. **다중 해상도 특징 적응형 융합(Multi-resolution Feature Adaptive Fusion) 모듈**: 스캐너 거리에 따른 밀도 차이를 극복하기 위해 다양한 해상도의 특징 맵을 적응적으로 융합하여 정밀하고 유의미한 특징 표현을 생성한다.
3. **상호 촉진형 공동 분할 프레임워크**: 의미론적 분할과 인스턴스 분할 브랜치가 서로의 학습된 특징을 주고받으며 성능을 향상시키는 통합 모델을 제안한다.

## 📎 Related Works

본 연구는 기존의 3D 포인트 클라우드 처리 방식인 PointNet, JSNet, JSPNet 등의 한계를 극복하고자 한다. 기존의 딥러닝 기반 방식들은 포인트 클라우드의 특성 추출에 어려움을 겪거나, 모델의 계산 및 메모리 소모가 크다는 단점이 있었다. 특히 셀프 어텐션(Self-attention) 메커니즘의 도입은 의미론적으로 풍부한 특징을 추출하여 성능을 높이는 방향으로 발전해 왔다.

본 논문은 단순한 특징 추출을 넘어, 의미론적 정보와 인스턴스 정보가 서로를 보완하도록 설계함으로써 기존의 개별 분할 방식이나 단순 결합 방식보다 정교한 결과를 도출한다.

## 🛠️ Methodology

### 1. Transformer Encoder-Decoder Module

이 모듈은 포인트 클라우드의 초기 특징을 추출하며, 크게 SA(Set Abstraction) 레이어와 Transformer 모듈로 구성된다.

* **SA Layer**: PointNet++의 구조를 따르되, 정보 손실을 줄이기 위해 최대 풀링(Maximum pooling) 대신 어텐션 풀링(Attention pooling)을 사용하여 유용한 정보를 자동으로 학습하고 집계한다.
* **Transformer Module**: 포인트 클라우드의 불규칙한 임베딩 특성을 처리하기 위해 Transformer 구조를 채택하였다. 위치 인코딩 $\delta$와 어텐션 벡터를 활용하며, 다음과 같은 수식으로 특징을 계산한다.

$$ \delta = \gamma(\text{pos}(P_i) - \text{pos}(P_j)) $$
$$ \text{Attention Calculation: } \text{Softmax}(\dots) \otimes (\text{Attention Vector} + \delta) $$

여기서 $\gamma(\cdot)$는 MLP(Multi-Layer Perceptron)이며, $\delta$는 포인트 간의 상대적 위치 정보를 제공하여 전역적 문맥을 파악하게 한다.

### 2. Multi-resolution Feature Adaptive Fusion Module

인코더를 통해 추출된 특징은 $N \times 512$ 차원의 행렬로 출력된다. 이후 두 개의 디코더 브랜치를 통해 업샘플링을 수행한다.

특히 인스턴스 브랜치에서는 정보의 추상화로 인한 손실을 막기 위해 4개의 서로 다른 해상도 특징 맵 $\{f'_1, f'_2, f'_3, f'_4\}$을 사용한다. 각 특징 맵에 대해 포인트 레벨의 인지값 $w$를 계산하고, 이를 통해 적응적으로 융합된 최종 특징 $S_{out}$을 생성한다.

$$ S_{out} = \sum_{i=1}^{4} \alpha_i f'_i $$

이 과정은 포인트 밀도가 낮은 영역에서도 강건한 특징을 유지하게 하여 인스턴스 분할의 정확도를 높인다.

### 3. Joint Instance and Semantic Segmentation Module

의미론적 분할과 인스턴스 분할 브랜치가 서로의 특징을 공유하여 성능을 높이는 구조이다. 단순히 정보를 병합하면 저품질의 정보가 유입될 수 있으므로, **Attentional Context Fusion** 모듈을 통해 가중 평균 방식으로 유용한 정보만 필터링하여 전달한다.

* **인스턴스 브랜치**: 의미론적 브랜치에서 온 특징을 퓨전 게이트(Fusion gate)를 통해 전달받아 인스턴스 임베딩 행렬 $F'_{ins}$를 생성한다. 최종적으로 Mean-shift clustering 알고리즘을 통해 개별 인스턴스를 구분한다.
* **의미론적 브랜치**: 인스턴스 브랜치에서 생성된 특징을 타일링(Tiling) 연산과 교차 평균(Cross-average)을 통해 의미론적 특징 공간에 통합하며, 최종 분류기를 통해 레이블을 결정한다.

### 4. Training 및 손실 함수

의미론적 분할은 교차 엔트로피 손실($L_{sem}$)을 사용하며, 인스턴스 분할은 다음과 같은 세 가지 요소의 합으로 구성된 손실 함수($L_{ins}$)를 사용한다.

$$ L_{ins} = L_{intra} + L_{inter} + L_{reg} $$

1. **Intra-instance similarity ($L_{intra}$)**: 동일 인스턴스 내 포인트들을 중심점으로 끌어당겨 유사성을 높인다.
2. **Inter-instance exclusivity ($L_{inter}$)**: 서로 다른 인스턴스 간의 임베딩을 멀어지게 하여 중복을 방지한다.
3. **Regularization ($L_{reg}$)**: 임베딩 값이 원점에서 너무 멀어지지 않도록 제한하여 클러스터 중심이 안정적으로 유지되게 한다.

최종 손실 함수는 $L = L_{sem} + L_{ins}$ 로 정의된다.

## 📊 Results

### 실험 설정

* **데이터셋**: S3DIS (Stanford Large-Scale 3D Indoor Spaces)
* **평가 지표**:
  * Semantic: overall Accuracy (oAcc), mean Accuracy (mAcc), mean IoU (mIoU)
  * Instance: mean Precision (mPre), mean Recall (mRec), mean Coverage (mCov), weighted Coverage (mWcov)
* **검증 방법**: 6-fold 교차 검증(CV) 및 특정 구역(Area 5)에 대한 별도 평가를 수행하여 공정성을 확보하였다.

### 정량적 결과

S3DIS 데이터셋의 Area 5 결과에서 본 모델은 기존 SOTA 모델들과 비교해 우수한 성능을 보였다.

* **Semantic Segmentation (Area 5)**:
  * mIoU 기준: **JSMNet(59.4%)** > JSPNet(56.1%) > JSNet(54.5%) > ASIS/PointNet(53.4%)
  * JSPNet 대비 mIoU가 3.3% 향상되었으며, PointNet 대비로는 16.0%의 압도적인 성능 향상을 보였다.
* **Instance Segmentation (Area 5)**:
  * mPre 기준: **JSMNet(59.9%)** > JSPNet(59.6%) > JSNet(62.1% - *텍스트 내 표 수치 참고 시 JSPNet보다 소폭 높거나 유사함*)
  * mCov 및 mWcov 지표에서도 JSPNet 대비 각각 0.7%씩 향상된 결과를 나타냈다.

### 정성적 결과

시각화 결과, 기존 JSNet이 테이블과 벽면을 명확히 구분하지 못하고 혼동하는 경향이 있었으나, JSMNet은 이를 명확하게 분리하여 분할하는 성능을 보여주었다.

## 🧠 Insights & Discussion

본 논문은 전역적 특징 추출(Transformer)과 지역적 특징 추출(PointConv)의 조화, 그리고 두 가지 분할 태스크 간의 상호 보완적 구조를 통해 실내 포인트 클라우드 분석 성능을 끌어올렸다. 특히 의미론적 정보가 인스턴스 분할의 가이드 역할을 하고, 인스턴스 정보가 다시 의미론적 경계를 명확히 하는 '상호 촉진(Mutual Promotion)' 구조가 유효했음을 입증하였다.

다만, 논문에서도 언급되었듯이 실내 환경의 심한 폐색(Occlusion), 복잡한 clutter, 가변성 문제는 여전히 해결해야 할 과제로 남아 있다. 또한, 본 연구는 제안된 모듈의 개별적 기여도에 대한 상세한 절제 연구(Ablation Study)가 텍스트상으로 충분히 제시되지 않아, 구체적으로 어떤 모듈이 가장 결정적인 성능 향상을 이끌었는지 파악하기 어렵다는 한계가 있다.

## 📌 TL;DR

본 논문은 Transformer의 글로벌 컨텍스트 추출 능력과 PointConv의 지역적 특징 추출을 결합하고, 다중 해상도 특징을 적응적으로 융합하는 **JSMNet**을 제안한다. 의미론적 분할과 인스턴스 분할 브랜치가 서로의 특징을 보완하는 상호 촉진 구조를 통해 S3DIS 데이터셋에서 기존 SOTA 모델(JSPNet 등)을 상회하는 성능을 달성하였다. 이 연구는 복잡한 실내 3D 장면 이해를 위한 효율적인 특징 융합 및 공동 학습 프레임워크를 제시했다는 점에서 향후 3D 씬 분석 연구에 중요한 기여를 할 것으로 보인다.
