# SurgPIS: Surgical-instrument-level Instances and Part-level Semantics for Weakly-supervised Part-aware Instance Segmentation

Meng Wei, Charlie Budd, Oluwatosin Alabi, Miaojing Shi, and Tom Vercauteren

## 🧩 Problem to Solve

로봇 보조 수술의 자동화를 위해 일관된 수술 도구 분할은 매우 중요합니다. 하지만 기존의 연구들은 도구 수준의 인스턴스 분할(IIS) 또는 부분 수준의 의미론적 분할(PSS)을 각각 독립적으로 다루었으며, 이 두 태스크 간의 상호작용은 고려하지 않았습니다. 또한, 수술 도구에 대한 인스턴스 및 부분 수준 레이블이 모두 포함된 대규모 데이터셋(Part-aware Instance Segmentation, PIS)이 부족하다는 문제가 있습니다.

## ✨ Key Contributions

* **최초의 수술 도구 PIS 모델 제안**: SurgPIS는 수술 도구를 위한 최초의 PIS(Part-aware Instance Segmentation) 모델로, 도구 인스턴스와 그 구성 요소를 동시에 식별합니다.
* **부분별 쿼리 변환 도입**: 기존 Mask2Former 모델을 확장하여, 도구 수준 객체 쿼리에서 파생된 부분별 쿼리(part-specific queries)를 도입함으로써 부분(part)과 상위 도구 인스턴스(parent instrument instances) 간의 계층적 연결을 명시적으로 구축합니다.
* **약지도 학습 전략 개발**: PIS 레이블이 없는 분리된 데이터셋(IIS 또는 PSS 레이블만 포함)을 활용하여 학습할 수 있는 새로운 약지도 학습(weakly-supervised learning) 전략을 제안합니다. 이는 데이터 부족 문제를 해결합니다.
* **최고 수준의 성능 달성**: PIS는 물론, IIS, PSS, 도구 수준 의미론적 분할(ISS) 등 다양한 분할 태스크에서 기존 최고 수준(state-of-the-art)의 성능을 뛰어넘습니다.
* **강력한 일반화 능력 입증**: 학습에 사용되지 않은 GraSP 데이터셋에 대한 평가를 통해 강력한 도메인 간 일반화 능력을 입증합니다.

## 📎 Related Works

* **수술 도구 분할**:
  * **IIS (Instance Segmentation)**: ISINet [15], TraSeTR [16], S3Net [17] 등은 개별 도구 인스턴스를 포착하는 데 중점을 둡니다.
  * **Semantic Segmentation (PSS/ISS)**: MF-TapNet [9], MATIS [25] 등은 도구 유형(ISS) 또는 부분 유형(PSS)에 대한 픽셀별 분류를 수행합니다.
  * **한계**: 이들 방법은 도구 인스턴스와 구성 부분 간의 명시적인 연결을 설정하지 않습니다.
* **Part-aware Panoptic Segmentation**:
  * 자연 이미지에서 객체 수준 세그먼트 내에 부분 수준 세그먼트를 통합하는 개념입니다 [26].
  * PartFormer [32], PartFormer++ [33]는 분리된 쿼리를 사용하며, TAPPS [20]는 공유 쿼리를 사용하지만 수술 도메인의 고유한 특성(유사한 도구 외형, 동일한 부분 유형)으로 인해 직접 적용 시 성능이 저하됩니다.
* **약지도 인스턴스 분할**:
  * 바운딩 박스(bounding box) [34]-[37], 자기 앙상블(self-ensembling) [38], 포인트 기반 감독(point-based supervision) [37] 등 다양한 약한 감독 신호를 활용합니다.
  * **한계**: 대부분의 기존 연구는 일관된 약한 감독 신호에 초점을 맞추며, 수술 도메인에서처럼 부분적으로만 레이블링된 불연속적인 데이터셋을 활용하는 접근 방식은 드뭅니다 [41].

## 🛠️ Methodology

SurgPIS는 두 단계의 학습 절차를 따릅니다.

1. **완전 지도 학습 (First Stage)**:
    * SurgPIS 모델은 PIS 레이블이 있는 데이터셋($D_{\text{PIS}}$)으로 완전 지도 방식으로 학습됩니다.
    * **SurgPIS 아키텍처**: Mask2Former [21]를 확장하여 설계되었습니다.
        * **백본 및 픽셀 디코더**: 멀티스케일 특징을 추출하여 고해상도 특징 맵 $F \in \mathbb{R}^{C_{\epsilon} \times H \times W}$를 생성합니다.
        * **도구 수준 쿼리**: $N_q$개의 학습 가능한 도구 수준 쿼리 $Q \in \mathbb{R}^{N_q \times C_{\epsilon}}$가 트랜스포머 디코더를 통해 도구 클래스 $\hat{c}_j$ 및 해당 인스턴스 마스크 $\hat{y}_j$를 예측합니다.
        * **부분별 쿼리 변환**: 핵심 모듈로, 도구 쿼리 $Q$가 다층 퍼셉트론(MLP)을 통해 부분별 쿼리 $Q_{\text{part}} \in \mathbb{R}^{(C_{\text{part}} \times N_q) \times C_{\epsilon}}$로 변환됩니다. 이는 부분별 바이너리 마스크 $\hat{m}_{j,k}$를 예측하는 데 사용되며, 부분과 상위 도구 간의 계층적 연결을 만듭니다.
        * **부분 인지 이분 매칭 (Part-aware Bipartite Matching)**: 예측과 정답 인스턴스 간의 매칭 시, 도구 수준 클래스 손실 $L_{\text{ic}}$, 도구 수준 마스크 손실 $L_{\text{im}}$ 외에 부분 수준 마스크 손실 $L_{\text{pm}}$을 포함한 비용 행렬을 구성합니다. 마스크 손실에는 focal loss와 Dice loss, 클래스 예측에는 Cross-entropy loss $\ell_{\text{CE}}$를 사용합니다.
        * **지도 손실**: 총 손실은 $L_{\text{sup}} = L_{\text{ic}} + L_{\text{im}} + L_{\text{pm}}$으로 정의됩니다.

2. **약지도 학습 (Second Stage)**:
    * $D_{\text{PIS}}$ 데이터를 계속 사용하면서, IIS 레이블만 있는 데이터셋($D_{\text{IIS}}$)과 PSS 레이블만 있는 데이터셋($D_{\text{PSS}}$)으로부터 약한 감독 신호를 도입합니다.
    * **Teacher-Student 모델**:
        * **초기화**: 1단계에서 학습된 SurgPIS 모델의 가중치로 학생 및 교사 모델을 초기화합니다.
        * **교사 모델 업데이트**: 교사 모델의 가중치 $\theta^{\text{teach}}$는 학습 중 학생 모델의 가중치 $\theta^{\text{stu}}$의 EMA(Exponential Moving Average)로 업데이트됩니다.
        * **데이터 증강**: 교사 모델 입력에는 약한 증강(random flipping)을 적용하고, 학생 모델 입력에는 강한 증강(color jitter, grayscale, Gaussian blur, patch erasing)을 적용합니다.
        * **가상 정답 필터링**: 교사 모델이 생성한 가상 정답 PIS 마스크 중 높은 신뢰도(Dice$(\hat{m}^{\text{teach}}_{\tau_i,k}, m_{i,k})_{i \neq 0} > T_{\text{Dice}}$)를 가진 마스크만 사용합니다.
        * **부분 의미론적 마스크 통합 (Part Semantic Mask Aggregation)**: $D_{\text{PSS}}$ 데이터의 경우, 학생 모델의 부분 수준 인스턴스 예측 $\hat{m}_{j,k}$를 도구 클래스 확률 $\hat{c}_j$로 가중하여 부분 의미론적 맵 $\hat{s}$으로 통합합니다. 이후 픽셀별 정규화를 통해 유효한 확률 분포를 만듭니다.
        * **약한 감독 손실**:
            * $D_{\text{PSS}}$의 경우: 통합된 부분 의미론적 맵 $\hat{s}$과 정답 $s$ 간의 $L^{\text{wks}}_{\text{pss}}$ (Dice loss).
            * $D_{\text{IIS}}$의 경우: 학생 도구 수준 출력과 정답 $y_{\text{iis}}$ 간의 $L^{\text{wks}}_{\text{iis}}$ (클래스 및 마스크 손실).
            * 총 약한 감독 손실: $L_{\text{wks}} = L^{\text{sup}}_{\text{teach}} + \begin{cases} L^{\text{wks}}_{\text{pss}} & \text{if PSS label} \\ L^{\text{wks}}_{\text{iis}} & \text{if IIS label} \end{cases}$

## 📊 Results

* **PIS 성능**:
  * 완전 지도 설정에서 SurgPIS는 EndoVis2018에서 11.07%p, EndoVis2017에서 11.23%p의 PartPQ 향상을 보여, 강력한 PIS 베이스라인(BPSS⊕BIIS)을 크게 능가합니다.
  * 약지도 설정(모든 데이터셋 사용)에서는 베이스라인 대비 EndoVis2018에서 14.49%p, EndoVis2017에서 13.65%p의 PartPQ 향상을 이룹니다.
  * 자연 이미지 PIS 모델인 TAPPS [20]보다 수술 도메인에서 월등히 높은 PartPQ 점수를 기록합니다.
* **IIS, PSS, ISS 성능**:
  * PIS를 위해 설계되었음에도 불구하고, SurgPIS는 PIS 예측을 집계하여 IIS, PSS, ISS 태스크에서도 SOTA 또는 매우 경쟁력 있는 성능을 달성합니다.
  * ISS 태스크에서 EndoVis2017의 Ch$_{\text{IoU}}$에서 S3Net [17] 대비 3.38%p, EndoVis2018에서 MATIS [25] 대비 3.6%p 향상을 보입니다.
* **일반화 성능 (GraSP 데이터셋)**:
  * 학습에 사용되지 않은 GraSP 데이터셋에서 테스트했을 때, SurgPIS는 IIS 태스크에서 기존 벤치마크 모델과 비교하여 PQ가 5.14%p, Ch$_{\text{IoU}}$가 3.95%p만 감소하며 강력한 일반화 능력을 입증합니다.
  * GraSP에서 PIS 태스크의 PartPQ는 TAPPS [20] 대비 32.85%p, BPSS⊕BIIS 대비 5.51%p 향상되었습니다.
* **Ablation Study**:
  * **부분 인지 이분 매칭 (PBM)**: PartPQ와 PartIoU에서 상당한 성능 저하(약 8~15%p)를 보여, 부분 수준 정보 통합의 중요성을 강조합니다.
  * **부분별 쿼리 변환 (PSQ)**: 이 모듈이 없을 경우 PartPQ가 약 35~40%p, PartIoU가 약 23%p 감소하여, 수술 도메인의 특수성(유사한 부분 유형)에 대한 맞춤형 쿼리의 중요성을 입증합니다.
  * **약지도 감독**: PSS 또는 IIS 감독을 제거하면 모든 지표에서 성능이 저하되며, 가상 레이블 필터링 및 강한 증강의 효과를 확인했습니다.
* **계산 효율성**: ResNet-50 백본 사용 시 30 FPS를 달성하여 실시간 적용에 적합함을 보여줍니다. 더 큰 백본(Swin-B, DINOv2-ViT-L)은 정확도를 높이지만 계산 비용이 증가합니다.

## 🧠 Insights & Discussion

* **통합 접근 방식의 시너지 효과**: 수술 도구 분할을 PIS 태스크로 재정의하고 공유 쿼리 메커니즘을 통해 계층적 표현을 학습하는 것은 멀티-세분성 분할을 가능하게 할 뿐만 아니라, 더 미세한 레이블을 통해 도구 수준 성능까지 향상시킵니다.
* **약지도 학습의 효과**: PIS 레이블 부족 문제를 해결하기 위한 약지도 학습 전략은 기존의 다양한, 부분적으로 레이블링된 데이터셋(IIS, PSS)을 성공적으로 활용하여 풍부한 특징 표현을 학습하고 지식 전이를 촉진합니다.
* **수술 도메인의 특수성 반영**: 자연 이미지 모델이 수술 도메인에서 실패하는 주요 원인 중 하나는 도구 간 유사한 외형과 동일한 부분 유형이 많다는 점입니다. SurgPIS의 부분별 쿼리 변환은 이러한 특수성을 효과적으로 다루는 핵심 요소입니다.
* **실시간 적용 가능성 및 확장성**: ResNet-50 백본을 사용하는 SurgPIS는 실시간 응용에 적합한 추론 속도를 제공합니다. 또한 다양한 백본, 특히 파운데이션 모델을 쉽게 통합할 수 있는 유연성을 보여주지만, 파운데이션 모델의 높은 계산 비용은 실시간 활용에 대한 트레이드오프를 가져옵니다.
* **향후 과제**: 현재는 분할 자체에 초점을 맞추고 있으며, 향후 실제 로봇 보조 수술에서의 잠재적인 응용(예: 로봇 제어, 고수준 수술 인지)을 탐색할 수 있습니다.

## 📌 TL;DR

SurgPIS는 수술 도구 분할을 위한 최초의 PIS(Part-aware Instance Segmentation) 모델로, 도구 인스턴스와 구성 부품을 동시에 식별합니다. 이 모델은 Mask2Former를 확장하여 계층적 연결을 위한 부분별 쿼리 변환을 도입하고, PIS 레이블이 부족한 수술 도메인의 특성을 고려하여 IIS 또는 PSS 레이블만 있는 분리된 데이터셋으로부터 학습할 수 있는 약지도 학습 전략(teacher-student 프레임워크 및 마스크 통합)을 제안합니다. SurgPIS는 PIS에서 SOTA를 달성할 뿐만 아니라, IIS, PSS, ISS 태스크에서도 뛰어난 성능과 강력한 일반화 능력을 보입니다.
