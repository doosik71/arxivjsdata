# DeSAM: Decoupled Segment Anything Model for Generalizable Medical Image Segmentation

Yifan Gao, Wei Xia, Dingdu Hu, Wenkui Wang, and Xin Gao (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델은 학습 데이터와 평가 데이터의 도메인이 동일할 때는 매우 높은 성능을 보이지만, 학습 시 보지 못한 새로운 도메인의 데이터가 입력될 때 성능이 급격히 떨어지는 도메인 시프트(Domain Shift) 문제에 취약하다. 이를 해결하기 위해 비지도 도메인 적응(Unsupervised Domain Adaptation)이나 다중 소스 도메인 일반화(Multi-source Domain Generalization) 방식이 제안되었으나, 이는 타겟 도메인의 데이터나 여러 소스 도메인의 데이터를 필요로 하며, 이는 실제 임상 환경에서 비용과 개인정보 보호 문제로 인해 적용하기 어렵다.

이에 따라 단일 소스 도메인 일반화(Single-source Domain Generalization, SSDG)가 더 실질적인 대안으로 떠오르고 있다. 최근 거대 파운데이션 모델인 Segment Anything Model (SAM)이 뛰어난 일반화 능력을 보여주며 의료 영상 분야에 적용되고 있으나, SAM은 수동 프롬프트(Manual Prompt)가 제공될 때에 비해 자동 분할(Automatic Segmentation) 시나리오에서 성능이 현저히 저하되는 한계가 있다. 본 논문은 이러한 성능 저하의 원인이 필연적으로 발생하는 부정확한 프롬프트와 마스크 생성 과정 사이의 결합 효과(Coupling Effect) 때문이라고 분석하며, 이를 해결하여 자동 분할 상황에서도 강건한 도메인 일반화 성능을 확보하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SAM의 마스크 디코더(Mask Decoder)를 두 개의 독립적인 하위 작업으로 분리하는 **Decoupled SAM (DeSAM)** 구조를 제안하는 것이다.

구체적으로는 프롬프트와 마스크 생성을 분리하여, 모델이 잘못된 프롬프트에 과도하게 의존하지 않도록 설계하였다. 이를 위해 프롬프트 관련 정보만을 처리하여 IoU 스코어와 마스크 임베딩을 생성하는 PRIM 모듈과, 이미지 인코더의 다중 스케일 특징과 PRIM의 임베딩을 융합하여 최종 마스크를 생성하는 PDMM 모듈을 도입하였다. 이러한 분리 설계를 통해 SAM의 사전 학습된 지식을 활용하면서도, 자동 분할 모드에서 발생하는 프롬프트 민감도 문제를 최소화하였다.

## 📎 Related Works

기존의 단일 소스 도메인 일반화(SSDG) 연구들은 주로 입력 공간 기반의 데이터 증강(예: Random Bias Field)이나 특징 공간 기반의 증강(예: RandConv, MaxStyle, CSDG)에 집중하였다. 그러나 이러한 방법들은 증강 함수를 설계하기 위한 전문가의 지식이 필요하거나 복잡한 적대적 학습(Adversarial Training) 과정이 수반되어야 한다는 한계가 있다.

한편, SAM을 의료 영상에 적응시키려는 최근의 시도들(MedSAM, SAMed 등)은 주로 마스크 디코더나 이미지 인코더를 미세 조정(Fine-tuning)하는 방식에 집중하였다. 하지만 이러한 접근법들은 특정 의료 작업으로의 적응에 초점을 맞추었을 뿐, 프롬프트 없이 수행되는 자동 분할 시나리오에서 발생하는 성능 저하 문제를 근본적으로 해결하지 못하였다. DeSAM은 구조적인 분리(Decoupling)를 통해 이 문제를 해결함으로써 기존 SAM 기반 적응 방식들과 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

DeSAM은 SAM의 이미지 인코더(Image Encoder)와 프롬프트 인코더(Prompt Encoder)를 계승하며, 기존의 마스크 디코더를 대신하여 **Prompt-Relevant IoU Module (PRIM)**과 **Prompt-Decoupled Mask Module (PDMM)**이라는 두 가지 새로운 모듈을 사용한다.

### 주요 구성 요소 및 역할

1. **Prompt-Relevant IoU Module (PRIM)**:
    * SAM의 마스크 디코더와 유사하게 Cross-attention Transformer 레이어와 IoU 예측 헤드로 구성된다.
    * 다만, 최종 마스크 예측 헤드를 제거하고 Cross-attention 레이어에서 생성된 **마스크 임베딩(Mask Embeddings)**만을 추출한다.
    * 역할은 주어진 프롬프트를 바탕으로 IoU 스코어를 예측하고, 마스크 생성을 위한 중간 임베딩을 제공하는 것이다.

2. **Prompt-Decoupled Mask Module (PDMM)**:
    * U-Net 및 UNETR의 인코더-디코더 구조에서 영감을 받아 설계되었다.
    * **다중 스케일 특징 추출**: SAM의 ViT-H 이미지 인코더의 글로벌 어텐션 레이어 $i \in \{8, 16, 24\}$에서 특징 맵을 추출한다. 이는 다양한 해상도의 계층적 표현을 캡처하여 풍부한 문맥 정보를 제공한다.
    * **특징 정제 및 융합**: Squeeze-and-Excitation (SE) residual blocks와 업샘플링(Upsampling) 연산을 통해 특징을 정제하며, 스킵 연결(Skip Connections)을 통해 저수준의 공간 세부 정보와 고수준의 의미 정보를 결합한다.
    * **최종 융합**: PDMM의 병목(Bottleneck) 임베딩에 PRIM에서 생성된 마스크 임베딩을 융합하여 최종 분할 마스크를 생성한다.

### 학습 절차 및 손실 함수

학습 시 SAM의 ViT-H 이미지 인코더와 프롬프트 인코더는 가중치를 고정(Freeze)하며, PRIM과 PDMM 레이어만 미세 조정한다. 자동 분할 방식에 따라 두 가지 학습 전략을 사용한다.

1. **Grid Points 모드**:
    * 정답 마스크 내부와 외부에 1:1 비율로 랜덤하게 포인트를 생성한다.
    * 마스크 생성은 Dice loss($L_{dice}$)와 Cross-entropy loss($L_{ce}$)로 감독하며, IoU 예측은 Mean Square Error loss($L_{mse}$)로 감독한다.
    * 전체 손실 함수는 다음과 같다.
    $$L_{points} = \lambda_1 L_{dice} + \lambda_2 L_{ce} + \lambda_3 L_{mse}$$
    * 이때 가중치는 $\lambda_1=1, \lambda_2=1, \lambda_3=10$으로 설정한다.

2. **Whole Box 모드**:
    * 정답 마스크가 반드시 박스 내부에 존재하므로, 마스크 생성에 대해서만 감독한다.
    $$L_{box} = L_{dice} + L_{ce}$$

## 📊 Results

### 실험 설정

* **데이터셋**: 교차 모달리티 복부 다기관 분할(Abdominal multi-organ, CT/MRI) 및 교차 사이트 전립선 분할(Prostate segmentation, 6개 임상 사이트 데이터)을 사용하였다.
* **비교 대상**: nnU-Net(Baseline 및 Upper bound), AdvBias, RandConv, MaxStyle, CSDG, MedSAM, SAMed.
* **평가 지표**: Dice Score.

### 주요 결과

* **정량적 성과**: DeSAM-P(Grid points 방식)가 전립선 분할 데이터셋에서 전체 평균 Dice score 79.02%를 기록하며 모든 비교 모델 중 가장 높은 성능을 보였다. 특히 기존 SOTA 방법인 CSDG보다 8.96% 향상된 수치를 기록하였다.
* **복부 데이터셋**: CT와 MRI 모달리티 모두에서 DeSAM-P와 DeSAM-B가 타 모델들을 일관되게 능가하였으며, DeSAM-P는 CT 86.68%, MRI 80.05%의 Dice score를 달성하여 새로운 SOTA 성능을 기록하였다.
* **소거 연구 (Ablation Study)**: PDMM 단독 사용보다 PRIM을 추가하고, IoU 예측 헤드(IPH)와 마스크 임베딩 융합(MEF)을 모두 적용했을 때 성능이 단계적으로 향상됨을 확인하였다.
* **프롬프트 강건성**: Grid points의 개수를 늘려도 성능 저하가 나타나지 않았으며, 이는 DeSAM이 잘못된 프롬프트로 인한 오탐(False Positive) 마스크 생성에 매우 강건함을 시사한다.

## 🧠 Insights & Discussion

DeSAM의 가장 큰 강점은 SAM의 강력한 일반화 능력을 유지하면서도, 자동 분할 시의 치명적 약점인 프롬프트 의존성을 구조적으로 해결했다는 점이다. 특히 PRIM과 PDMM으로 역할을 분리함으로써, 모델이 프롬프트에 휘둘리지 않고 이미지 자체의 강건한 특징(Prompt-invariant features)을 학습하도록 유도하였다.

또한, 이미지 인코더의 서로 다른 레이어(8, 16, 24)에서 특징을 추출하여 융합하는 다중 스케일 전략은 의료 영상에서 매우 중요한 세밀한 경계 정보와 전역적인 문맥 정보를 동시에 포착하는 데 기여하였다.

다만, 본 연구는 이미지 인코더의 가중치를 고정하고 디코더 부분만 수정하는 방식을 취했다. 만약 이미지 인코더의 일부 레이어를 도메인에 맞게 적응시키는 어댑터(Adapter) 구조를 추가한다면, 성능이 더욱 향상될 가능성이 있다. 또한, 사용된 데이터셋의 특성에 따라 최적의 Grid point 개수나 하이퍼파라미터가 다를 수 있다는 점이 향후 고려 사항이다.

## 📌 TL;DR

본 논문은 SAM의 자동 분할 성능을 저하시키는 '결합 효과(Coupling Effect)'를 해결하기 위해 프롬프트 처리(PRIM)와 마스크 생성(PDMM)을 분리한 **DeSAM**을 제안한다. 이를 통해 단일 소스 도메인 일반화(SSDG) 환경에서 기존 SOTA 모델들을 크게 상회하는 강건한 의료 영상 분할 성능을 입증하였다. 이 연구는 파운데이션 모델을 의료 분야에 적용할 때, 단순한 미세 조정을 넘어 구조적 분리를 통해 도메인 일반화 능력을 극대화할 수 있음을 보여주었으며, 향후 다양한 의료 영상 자동 분할 시스템의 기초 구조로 활용될 가능성이 높다.
