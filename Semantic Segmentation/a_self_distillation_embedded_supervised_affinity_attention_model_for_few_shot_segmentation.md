# A Self-Distillation Embedded Supervised Affinity Attention Model for Few-Shot Segmentation

Qi Zhao, Binghao Liu, Shuchang Lyu, and Huojin Chen

## 🧩 Problem to Solve

본 논문은 제한된 주석이 달린 샘플만으로 이전에 보지 못한 객체를 분할하는 Few-Shot Segmentation 태스크가 직면한 두 가지 주요 과제를 해결하고자 합니다.

1. **지원(Support) 및 쿼리(Query) 이미지 간의 엄청난 특징(Feature) 차이**: 동일 범주의 객체라도 촬영 각도, 조명 조건, 모양 및 색상 등의 차이로 인해 특징 간의 큰 불일치가 발생하며, 이는 지식 전달을 방해하여 분할 성능을 저해합니다.
2. **제한된 지원 프로토타입(Prototype)의 불충분한 정보 제공**: 소수의 지원 프로토타입으로는 지원 객체의 특징을 충분히 대표하기 어려워, 고품질 쿼리 분할을 가이드하는 데 한계가 있습니다.

## ✨ Key Contributions

* **Self-Distillation Guided Prototype Module (SDPM) 제안**: 자기 증류(self-distillation) 방식을 사용하여 지원 및 쿼리 특징 간의 격차를 해소하고 본질적인(intrinsic) 클래스 특징을 가진 프로토타입을 효율적으로 정렬 및 생성합니다.
* **Supervised Affinity Attention Module (SAAM) 제안**: CNN 기반의 엔드-투-엔드 모듈로, 지원 마스크(ground truth)의 감독을 통해 고품질의 쿼리 어텐션 맵(attention map)을 생성하여 디코더가 집중해야 할 영역을 명확히 알려줍니다.
* **SD-AANet (Self-Distillation embedded Affinity Attention Network) 모델 구축**: SDPM과 SAAM을 결합하여 위에서 언급된 두 가지 주요 Few-shot Segmentation 과제를 효과적으로 해결합니다.
* **최첨단 성능 달성**:
  * COCO-20${_i}$ 데이터셋에서 새로운 SOTA(State-of-the-Art) 결과를 달성했습니다 (1-shot mIoU: 40.9%, 5-shot mIoU: 48.7%).
  * PASCAL-5${_i}$ 데이터셋에서 비교할 만한 SOTA 결과를 달성했습니다 (1-shot mIoU: 62.9%).
* **다중 클래스 Few-shot Segmentation을 위한 새로운 파이프라인 제안**: SD-AANet의 잠재력을 확장하여 여러 클래스의 객체를 동시에 분할할 수 있는 방법을 제시합니다.

## 📎 Related Works

* **Semantic Segmentation**: FCN [1], SegNet [2], UNet [3], PSPNet [6], DeepLab [5], [19], [20]와 같은 심층 신경망 모델 및 PSANet [21], DANet [22], CCNet [23]과 같은 어텐션 메커니즘을 활용한 연구들이 언급됩니다.
* **Few-shot Learning**: Metric learning(Siamese Network [24], Matching Networks [25]) 및 Meta-learning(Meta-learning LSTM [28], MAML [29], ProtoMAML [30]) 방식이 주로 다루어집니다.
* **Few-shot Segmentation**: OSLSM [8]을 시작으로 SG-One [12], PANet [31], PFENet [34], ASNet [37] 등 프로토타입 학습 및 어피니티 학습에 기반한 다양한 선행 연구들이 참고됩니다.
* **Knowledge Distillation (지식 증류) / Self-Distillation (자기 증류)**: Hinton et al. [41]의 초기 연구부터 Zagoruyko et al. [42], He et al. [43], Lyu et al. [45] 등 단일 신경망 내에서의 지식 증류에 대한 연구에 영감을 받았습니다.

## 🛠️ Methodology

본 논문은 `SD-AANet` (Self-Distillation embedded Affinity Attention Network)이라는 새로운 Few-shot Segmentation 모델을 제안합니다.

* **전체 아키텍처 (그림 3 참조)**:
  * 공유 백본 CNN(ResNet50/101)을 통해 지원 이미지($I_s$)와 쿼리 이미지($I_q$)로부터 지원 특징($F_s$) 및 쿼리 특징($F_q$)을 추출합니다.
  * 이 특징들과 지원 마스크($M_s$)는 `SDPM`과 `SAAM`의 입력으로 사용됩니다.
  * `SDPM`은 채널 재가중 쿼리 특징($\tilde{F}_q$)과 본질적 지원 프로토타입을 출력합니다.
  * `SAAM`은 어피니티 어텐션 맵을 생성합니다.
  * 이 모든 정보를 결합하여 디코더에 입력하고, 최종 분할 예측을 출력합니다.
  * 총 손실 함수는 분할 손실, 자기 증류 손실, SAAM의 교차 엔트로피 손실의 가중합으로 구성됩니다: $L = L_{ce} + \alpha \cdot L_{KD} + \beta \cdot L_{ce,s}$.

* **Self-Distillation Guided Prototype Generating Module (SDPM) (그림 4 참조)**:
    1. **지원 기반 채널 재가중(Support-Guided Channel Reweighting)**:
        * 지원 특징($F_s$)과 지원 마스크($M_s$)를 사용하여 마스킹된 GAP(Global Average Pooling)를 통해 초기 지원 프로토타입($p_s$)을 생성합니다.
        * $p_s$는 FC 계층을 통과하여 채널 재가중 벡터($v_s$)를 생성합니다.
        * 이 $v_s$는 지원 특징($F_s$)과 쿼리 특징($F_q$)의 채널에 적용되어 재가중된 특징($\tilde{F}_s$, $\tilde{F}_q$)을 만듭니다. 이는 스케일링된 특징과 원본 특징의 평균을 취하는 퓨전 전략을 사용합니다.
    2. **자기 증류 임베디드 방식(Self-distillation embedded Method)**:
        * 재가중된 지원 특징($\tilde{F}_s$)과 쿼리 특징($\tilde{F}_q$)으로부터 각각 지원 프로토타입($p'_s$)과 쿼리 프로토타입($p'_q$)을 생성합니다. ($M_q$는 쿼리 프로토타입 생성에만 사용되며, 최종 예측의 GT로 직접 사용되지는 않습니다.)
        * 교사 프로토타입($d_t$)은 $p'_s$와 $p'_q$의 softmax 출력 평균으로 정의됩니다: $d_t = \frac{\text{Softmax}(p'_{s}) + \text{Softmax}(p'_{q})}{2}$.
        * KL 발산 손실($L_{KD} = \text{KL}(d_t \parallel \text{Softmax}(p'_s))$)을 사용하여 $p'_s$를 $d_t$에 정렬시킴으로써 본질적인 특징을 강화하고 고유한 특징을 줄입니다.
    3. **K-shot 설정 (그림 5 참조)**:
        * **통합 교사 프로토타입 전략(Integral Teacher Prototype Strategy)**: K개의 지원 샘플의 재가중 벡터 평균을 쿼리 특징에 적용하고, K개의 개별 지원 프로토타입 각각에 대해 하나의 쿼리 프로토타입을 교사로 사용하여 KD 손실을 계산합니다.
        * **개별 교사 프로토타입 전략(Separate Teacher Prototype Strategy)**: 각 지원 샘플의 재가중 벡터를 쿼리 특징에 개별적으로 적용하여 K개의 개별 교사 프로토타입을 생성하고, 각 지원 프로토타입에 대해 해당 교사를 사용하여 KD 손실을 계산합니다. (실험 결과, 개별 교사 전략이 더 우수하여 최종 모델에 채택됩니다.)
        * 최종 출력 지원 프로토타입은 K개 지원 프로토타입의 평균입니다.

* **Supervised Affinity Attention Module (SAAM) (그림 6 참조)**:
    1. 마스킹된 GAP를 통해 지원 프로토타입을 얻고, 이를 지원 특징($F_s$) 및 쿼리 특징($F_q$)과 공간적으로 동일한 크기로 확장하여 각각 $F_{C,s}$와 $F_{C,q}$를 만듭니다.
    2. 이들을 Pyramid Pooling Module (PPM)에 입력합니다.
    3. PPM 출력 후 두 개의 $1 \times 1$ 컨볼루션 계층을 통해 다음을 생성합니다:
        * 지원 예측 (2 채널): 지원 마스크($M_s$)와 교차 엔트로피 손실($L_{ce,s}$)로 감독됩니다.
        * 쿼리 어텐션 맵 (1 채널): 최종 분할 예측을 위한 가이드로 사용됩니다.
    4. 이러한 감독 방식은 쿼리 어텐션 맵의 품질을 향상시킵니다.
    5. **K-shot 설정**: K개의 지원 특징이 SAAM을 개별적으로 통과하며, 각 지원 예측은 해당 마스크로 감독됩니다. K-shot SAAM 손실은 개별 손실의 평균입니다.

* **다중 클래스 Few-shot Segmentation (알고리즘 1 참조)**:
  * 입력은 쿼리 이미지 및 마스크와 함께 5개의 서로 다른 클래스에서 온 5개의 지원 이미지 및 마스크입니다.
  * SDPM과 SAAM을 거친 후, 하나의 쿼리 특징, 5개의 지원 프로토타입, 5개의 어텐션 맵이 생성됩니다.
  * 5개의 지원 프로토타입은 하나의 벡터로 연결된 후 MLP(Multi-Layer Perceptron)를 통해 차원이 축소됩니다.
  * 이 축소된 벡터와 쿼리 특징, 5개의 어텐션 맵을 디코더의 입력으로 연결합니다.
  * 디코더의 최종 출력 채널은 2에서 5로 변경되어 5개 클래스의 객체를 동시에 분할할 수 있도록 합니다.

## 📊 Results

* **PASCAL-5${_i}$ 데이터셋 (표 I 참조)**:
  * SD-AANet (ResNet50 백본)은 1-shot에서 62.9%, 5-shot에서 65.5%의 mIoU를 달성하여 PFENet 기반의 Baseline 모델(1-shot 60.4%) 대비 상당한 성능 향상을 보이며 SOTA에 비견되는 결과를 얻었습니다.
  * 특히 Fold-1 1-shot 태스크에서 우수한 성능을 보여주었습니다.

* **COCO-20${_i}$ 데이터셋 (표 II 참조)**:
  * SD-AANet (ResNet101 백본)은 1-shot에서 40.9%, 5-shot에서 48.7%의 mIoU를 달성하여 새로운 SOTA를 기록했으며, CMN [57]보다 1-shot에서 1.6%, 5-shot에서 5.6% 우수합니다. Baseline 대비 각각 2.4%, 3.7%의 성능 향상을 보였습니다.
  * 입력 이미지 크기를 이전 연구들보다 작은 $321 \times 321$로 사용했음에도 불구하고 뛰어난 성능을 달성했습니다.

* **복잡도 및 계산 효율성 (표 III 참조)**:
  * SD-AANet은 Baseline 모델에 비해 GPU 메모리 사용량이 12%, 학습 가능한 파라미터 수가 30% 증가했지만, 추론 속도(FPS)는 18.75에서 17.65로 약간 감소하는 정도에 그쳐 효율성을 유지했습니다.

* **어블레이션 연구 (Ablation Study)**:
  * **SDPM과 SAAM의 효과 (표 IV 참조)**: Baseline 대비 SAAM 단독 사용 시 mIoU 1.2% 증가, SDPM 단독 사용 시 1.6% 증가, 둘 다 사용 시 2.5% 증가를 보여 각 모듈의 효과와 시너지 효과를 확인했습니다.
  * **다중 스케일 추론 (표 V 참조)**: 단일 스케일 추론($62.9\%$ mIoU)에 비해 다중 스케일 추론($63.2\%$ mIoU)이 약간의 성능 향상(0.3% mIoU)을 가져왔습니다.
  * **5-shot 전략 (표 VI 참조)**: SDPM의 K-shot 전략 중 `Separate Teacher Prototype Strategy`가 `Integral Teacher Prototype Strategy`보다 더 나은 성능을 보여주었습니다 (65.5% vs 64.6% mIoU).
  * **다중 클래스 분할 (표 VII, VIII 참조)**: 다중 클래스 1-shot 분할 태스크에서 SD-AANet은 Baseline 대비 평균 3.9% mIoU 증가를 달성했습니다. 특히 'motorbike'(Fold-2, Class 4)와 같은 어려운 클래스에서 현저한 개선을 보였습니다.

## 🧠 Insights & Discussion

* **SDPM의 효과 (t-SNE 시각화, 그림 8 참조)**: SDPM은 클래스 간의 프로토타입 거리를 크게 확장하고 동일 클래스 내 프로토타입을 더 밀집하게 만들어, 지원 및 쿼리 특징을 효과적으로 정렬하고 본질적인 특징을 추출하는 데 기여합니다. 이는 외형적 차이가 큰 경우에도 뛰어난 분할 성능으로 이어집니다.
* **SAAM의 효과 (어피니티 어텐션 시각화, 그림 10 참조)**: SAAM은 작은 객체나 여러 객체가 있는 경우에도 대상의 공간 정보를 효과적으로 포착하고, 버스의 백미러나 비행기의 바퀴와 같이 분리된 부분의 핵심 정보를 집중적으로 파악하여 고품질의 어텐션 맵을 생성합니다.
* **작은 객체에 대한 성능 (그림 9 참조)**: SD-AANet은 5000픽셀 미만의 작은 객체 분할에서 Baseline 모델보다 일관되게 높은 mIoU를 달성하여 작은 객체에 대한 우수한 성능을 입증했습니다.
* **프로토타입의 대표성 (유사성 맵 시각화, 그림 11 참조)**: SD-AANet에 의해 생성된 지원 프로토타입은 Baseline에 비해 불필요한 배경을 더 많이 필터링하고 대상의 전체 공간 영역을 포착하는 능력을 보여, 환경적 요인에 관계없이 본질적인 특징에 집중함을 나타냅니다.
* **한계점 (실패 사례 시각화, 그림 12 참조)**: SD-AANet은 여전히 대상 크기가 너무 작거나 배경과 매우 유사하여 구별하기 어려운 경우에는 분할에 실패하는 경우가 있습니다.
* **향후 연구 방향**: 자기 증류를 제로샷(zero-shot) 분할 태스크에 활용하는 방안을 모색할 계획입니다.

## 📌 TL;DR

* **문제**: Few-shot Segmentation은 지원-쿼리 이미지 간의 큰 특징 차이와 제한적인 지원 프로토타입 표현력이라는 두 가지 주요 어려움을 겪습니다.
* **해결책**: 본 논문은 `SD-AANet`이라는 새로운 모델을 제안하며, 이는 `SDPM` (자기 증류를 통해 본질적인 프로토타입을 추출하여 특징을 정렬)과 `SAAM` (감독 학습을 통해 고품질 어텐션 맵 생성)을 통합합니다.
* **결과**: `SD-AANet`은 COCO-20${_i}$에서 새로운 SOTA를, PASCAL-5${_i}$에서 비교할 만한 SOTA를 달성하여, 특징 불일치를 효과적으로 해소하고 풍부한 대상 정보를 제공함으로써 Few-shot Segmentation 성능을 크게 향상시켰음을 입증합니다.
