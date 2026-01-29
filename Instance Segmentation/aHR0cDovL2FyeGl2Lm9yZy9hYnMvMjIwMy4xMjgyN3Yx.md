# Sparse Instance Activation for Real-Time Instance Segmentation

Tianheng Cheng, Xinggang Wang, Shaoyu Chen, Wenqiang Zhang, Qian Zhang, Chang Huang, Zhaoxiang Zhang, Wenyu Liu

## 🧩 Problem to Solve

기존의 인스턴스 분할(Instance Segmentation) 방법들은 대부분 객체 탐지(Object Detection)에 크게 의존하며, 바운딩 박스나 밀집된 중심(dense centers) 기반으로 마스크 예측을 수행합니다. 이러한 방식은 다음과 같은 문제점을 가집니다:

* **과도한 예측 및 계산 부담:** 수많은 앵커나 중심점을 사용하여 객체를 지역화하므로 중복된 예측과 높은 계산량을 야기합니다.
* **제한된 수용장(Receptive Field) 및 문맥 정보 부족:** 각 픽셀의 수용장이 제한되어 밀집된 방식으로 객체를 지역화할 경우 문맥 정보가 불충분합니다.
* **다단계 예측의 지연 시간 증가:** 다양한 스케일의 객체 처리를 위해 다단계 예측을 사용하면 필연적으로 지연 시간이 증가합니다. 특히 RoI-Align 같은 연산은 엣지 디바이스에 배포하기 어렵게 만듭니다.
* **느린 후처리(Post-processing):** 정렬 및 NMS(Non-Maximum Suppression), 마스크 처리 등이 시간을 많이 소모하며, 특히 밀집된 예측에서는 더욱 심각합니다.

실시간 인스턴스 분할, 특히 자율주행 및 로봇 공학 분야에서 이러한 문제점을 해결하고 효율적인 알고리즘을 개발하는 것이 시급합니다.

## ✨ Key Contributions

* **새로운 객체 표현 방식 제안:** 객체 감지 및 분할을 위해 바운딩 박스나 중심점 대신 `스파스 인스턴스 활성화 맵(Sparse Instance Activation Maps, IAM)`이라는 개념적으로 새로운 객체 표현 방식을 제안합니다. IAM은 각 전경 객체에 대한 정보성 영역을 강조합니다.
* **IAM 기반 특징 집계 및 NMS 제거:** 강조된 영역에 따라 인스턴스 수준 특징을 직접 집계하여 인식 및 분할을 수행합니다. 또한, 이분 매칭(bipartite matching)을 기반으로 IAM이 객체를 1:1 방식으로 예측하여 후처리에서 NMS를 피할 수 있습니다.
* **효율적인 완전 컨볼루션 프레임워크 SparseInst 개발:** IAM을 활용한 간단하면서도 효과적인 설계로 `SparseInst`를 제안합니다. 이는 탐지기에 독립적인 순수하고 완전 컨볼루션 프레임워크입니다.
* **최고 수준의 속도 및 정확도 달성:** MS-COCO 벤치마크에서 40 FPS 및 37.9 AP를 달성하여 기존 실시간 인스턴스 분할 방법들보다 속도와 정확도 면에서 크게 우수함을 입증했습니다. (예: 448x448 입력에서 58.5 FPS, 35.5 AP)
* **단일 레벨 예측 및 경량화된 구조:** 스파스 예측, 단일 레벨 예측, 경량 구조, NMS 없는 간단한 후처리 덕분에 매우 빠른 추론 속도를 가집니다.

## 📎 Related Works

* **Region-based Methods:** Faster R-CNN, Mask R-CNN과 같이 객체 탐지기를 사용하여 바운딩 박스를 얻은 후 RoI-Pooling 또는 RoI-Align을 통해 특징을 추출하여 픽셀 단위 분할을 수행하는 방법들. (예: Mask R-CNN, Cascade R-CNN)
* **Center-based Methods:** FCOS와 같은 앵커 프리(anchor-free) 탐지기를 활용하여 중심 픽셀로 객체를 표현하고 중심 특징을 사용하여 분할하는 방법들. (예: YOLACT, MEInst, CondInst, SOLO/SOLOv2, PolarMask)
* **Bipartite Matching for Object Detection:** DETR에서 시작하여 NMS를 피하기 위해 이분 매칭을 활용하는 엔드투엔드 객체 탐지/분할 방법들. (예: SOLQ, ISTR, QueryInst, Deformable DETR)

## 🛠️ Methodology

SparseInst는 백본 네트워크, 인스턴스 문맥 인코더, IAM 기반 디코더의 세 가지 주요 구성 요소로 이루어진 단순하고 효율적인 완전 컨볼루션 프레임워크입니다.

1. **인스턴스 활성화 맵(Instance Activation Maps, IAM):**
    * 객체별로 정보성 영역을 강조하는 가중치 맵 $A = F_{\text{iam}}(X) \in \mathbb{R}^{N \times (H \times W)}$ 입니다.
    * 강조된 영역의 특징을 집계하여 인스턴스 특징 $z = \bar{A} \cdot X^T \in \mathbb{R}^{N \times D}$를 얻습니다.
    * 명시적인 마스크 감독 없이 인식 및 분할 모듈의 간접적인 감독과 이분 매칭을 통해 학습됩니다.
    * `Group-IAM`은 각 객체에 대해 여러 개의 활성화 맵을 생성하여 더 세밀한 특징을 얻습니다.
2. **인스턴스 문맥 인코더(Instance Context Encoder):**
    * 백본(예: ResNet)에서 추출된 다중 스케일 특징 $\{C_3, C_4, C_5\}$를 받아들입니다.
    * $C_5$ 이후에 `피라미드 풀링 모듈(PPM)`을 적용하여 수용장(receptive fields)을 확장합니다.
    * P3에서 P5까지의 다중 스케일 특징을 융합하여 단일 레벨 특징 표현을 강화합니다 (입력 이미지 대비 $1/8$ 해상도).
3. **IAM 기반 디코더(IAM-based Decoder):**
    * **위치 민감 특징(Location-Sensitive Features):** 인코더의 출력 특징에 정규화된 $(x,y)$ 좌표 특징을 연결하여 인스턴스 인식 표현을 강화합니다.
    * **인스턴스 브랜치(Instance Branch):**
        * $3 \times 3$ 컨볼루션(Sigmoid 활성화)을 사용하여 IAM을 예측합니다.
        * IAM을 통해 얻은 인스턴스 특징 $z_i$를 사용하여 분류, 객체성(objectness) 점수, 마스크 커널 $w_i$를 예측합니다.
        * `IoU-aware Objectness`: 예측 마스크와 GT 마스크 간의 IoU를 객체성 예측의 목표로 사용하여 분류 출력의 신뢰도를 조정합니다. 추론 시 최종 확률 $\tilde{p}_i = \sqrt{p_i \cdot s_i}$로 재조정합니다.
    * **마스크 브랜치(Mask Branch):**
        * 인스턴스별 마스크 특징 $M$을 제공합니다.
    * **마스크 생성:** 각 인스턴스에 대한 분할 마스크 $m_i = w_i \cdot M$는 인스턴스 브랜치에서 생성된 마스크 커널과 마스크 브랜치의 특징을 곱하여 생성됩니다.
4. **레이블 할당 및 이분 매칭 손실(Label Assignment and Bipartite Matching Loss):**
    * 고정된 수의 예측과 GT 객체 간의 매칭을 위해 이분 매칭 문제를 정식화합니다.
    * **매칭 점수:** $C(i,k) = p_{i,c_k}^{1-\alpha} \cdot \text{DICE}(m_i, t_k)^\alpha$ (여기서 $\alpha=0.8$)를 사용하여 $i$-번째 예측과 $k$-번째 GT 객체 간의 최적 매칭을 `헝가리안 알고리즘(Hungarian algorithm)`으로 찾습니다.
    * **훈련 손실:** 분류를 위한 Focal Loss ($L_{cls}$), 마스크 손실 ($L_{mask}$), IoU-aware 객체성을 위한 이진 교차 엔트로피 손실 ($L_s$)로 구성됩니다: $L = \lambda_c \cdot L_{cls} + L_{mask} + \lambda_s \cdot L_s$.
    * **하이브리드 마스크 손실:** $L_{mask} = \lambda_{dice} \cdot L_{dice} + \lambda_{pix} \cdot L_{pix}$로, 다이스 손실(Dice Loss)과 픽셀 단위 이진 교차 엔트로피 손실(Binary Cross Entropy Loss)을 결합하여 전경/배경 불균형 문제를 해결합니다.
5. **추론(Inference):**
    * 전체 네트워크를 통해 이미지를 순전파하여 분류 점수와 마스크를 직접 얻습니다.
    * 임계값(thresholding)을 적용하여 최종 이진 마스크를 생성하며, 정렬이나 NMS가 필요 없어 매우 빠릅니다.

## 📊 Results

* **속도-정확도 최적화:** COCO test-dev 벤치마크에서 SOTA(State-of-the-Art) 실시간 방법들을 능가하는 속도와 정확도 트레이드오프를 보여줍니다.
  * ResNet-50-d-DCN 백본, 608x608 입력 시: 40.0 FPS, Mask AP 37.9
  * ResNet-50-d-DCN 백본, 448x448 입력 시: 58.5 FPS, Mask AP 35.5
  * YOLACT++보다 더 높은 AP와 더 빠른 FPS를 달성합니다.
* **구성 요소별 효과 (Ablation Study):**
  * **인스턴스 문맥 인코더:** PPM 및 다중 스케일 융합을 통해 AP를 크게 향상시키며, 특히 큰 객체($\text{AP}_\text{L}$)에서 효과적입니다 (1.5 AP 이상 증가). 추가 지연 시간은 미미합니다.
  * **디코더 구조:** 좌표 특징 추가 시 0.5 AP 향상. DCN(Deformable Convolution) 사용 시 큰 객체($\text{AP}_\text{L}$)에서 상당한 개선을 보이지만, 추론 시간은 증가합니다.
  * **인스턴스 활성화 맵($F_{iam}$):** Group-IAM (4 그룹) 사용 시 0.7 AP 개선을 달성합니다. Softmax나 $1 \times 1$ 컨볼루션보다 $3 \times 3$ 컨볼루션(Sigmoid)이 우수합니다.
  * **하이브리드 마스크 손실:** Dice Loss가 마스크 예측에 필수적이며 (제거 시 AP 8.1p 하락), 픽셀 단위 BCE Loss를 추가하면 AP 1.0p 향상, 특히 큰 객체($\text{AP}_\text{L}$)에서 효과적입니다.
  * **IoU-aware Objectness:** 1.3 AP를 향상시키며, 재점수화(rescoring)하지 않아도 0.7 AP 향상을 가져와 인스턴스 브랜치가 더 인스턴스 인식 특징을 학습하도록 돕습니다.
* **추론 시간 분석:** 백본(ResNet-50)이 전체 추론 시간의 50% 이상을 차지하며, 후처리도 약 2ms가 소요됩니다.

## 🧠 Insights & Discussion

* **IAM의 효과성:** 인스턴스 활성화 맵(IAM)은 객체별로 차별화된 영역을 강조하여, 기존의 중심 기반 또는 영역 기반 방법의 단점(잘못된 특징 지역화, 주변 객체/배경 특징 포함)을 개념적으로 피할 수 있습니다. 이는 RoI-Align과 같은 복잡한 연산 없이 이미지 전체에서 문맥 정보를 집계하여 인스턴스 특징을 추출하는 이점을 제공합니다.
* **엔드투엔드 및 NMS 없는 추론:** 이분 매칭을 통한 레이블 할당은 SparseInst가 중복 예측 없이 엔드투엔드 방식으로 훈련되도록 하여, 복잡하고 시간 소모적인 NMS 후처리를 제거합니다. 이는 실시간 성능 달성에 핵심적인 역할을 합니다.
* **속도-정확도 트레이드오프의 우수성:** 단일 레벨 예측, 스파스 예측, 간결한 구조, 단순화된 후처리 덕분에 SparseInst는 높은 정확도를 유지하면서도 매우 빠른 추론 속도를 달성합니다. 이는 특히 자율주행과 같은 지연 시간에 민감한 애플리케이션에 매우 유리합니다.
* **한계점 및 향후 방향:** 현재 백본 네트워크가 추론 시간의 대부분을 차지하므로, 더 효율적인 백본 사용이나 디코더의 $3 \times 3$ 컨볼루션 가지치기 등을 통해 추가적인 속도 최적화가 가능할 것으로 보입니다.
* **범용 프레임워크 가능성:** SparseInst의 단순하고 효율적인 설계는 엔드투엔드 실시간 인스턴스 분할을 위한 범용 프레임워크로 활용될 잠재력을 가지고 있습니다.

## 📌 TL;DR

실시간 인스턴스 분할의 복잡성과 NMS 기반 후처리 문제를 해결하기 위해, 이 논문은 **스파스 인스턴스 활성화 맵(Sparse Instance Activation Maps, IAM)**이라는 새로운 객체 표현 방식을 제안합니다. 이 IAM은 객체별 특징 영역을 직접 강조하고, **이분 매칭**을 통해 NMS 없는 엔드투엔드 훈련 및 추론을 가능하게 합니다. 그 결과, **SparseInst**는 COCO 벤치마크에서 40 FPS, 37.9 AP를 달성하며 기존 SOTA 실시간 방법들보다 뛰어난 속도-정확도 트레이드오프를 보여주었습니다.
