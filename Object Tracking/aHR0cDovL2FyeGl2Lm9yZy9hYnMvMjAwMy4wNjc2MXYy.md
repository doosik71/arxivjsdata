# Siamese Box Adaptive Network for Visual Tracking
Zedu Chen, Bineng Zhong, Guorong Li, Shengping Zhang, Rongrong Ji

## 🧩 Problem to Solve
기존 시각 객체 추적기들은 대개 대상의 크기(scale)와 종횡비(aspect ratio)를 정확하게 추정하기 위해 다중 스케일 탐색(multi-scale searching) 방식이나 사전 정의된 앵커 박스(pre-defined anchor boxes)에 의존합니다. 이러한 방식들은 종종 번거롭고 경험적인 설정(heuristic configurations)을 필요로 하여 유연성과 일반성이 떨어집니다. 이 논문은 이러한 비효율적인 설정을 피하면서 대상의 크기와 종횡비를 정확하게 추정하는 문제를 해결하고자 합니다.

## ✨ Key Contributions
*   **앵커 프리(Anchor-free) 설계:** 사전 정의된 후보 박스(candidate boxes)나 앵커 박스 없이 시각 객체 추적을 수행하는 Siamese Box Adaptive Network (SiamBAN) 프레임워크를 제안합니다. 이로써 하이퍼파라미터 설정의 번거로움을 피하고 추적기의 유연성 및 일반성을 향상시킵니다.
*   **통합된 FCN 기반 분류-회귀:** 추적 문제를 단일의 완전 컨볼루션 네트워크(Fully Convolutional Network, FCN) 내에서 병렬적인 분류(classification) 및 경계 박스 회귀(bounding box regression) 문제로 변환하여 해결합니다.
*   **고성능 및 효율성:** VOT2018, VOT2019, OTB100, NFS, UAV123, LaSOT 등 다양한 시각 객체 추적 벤치마크에서 최첨단 성능을 달성하면서도 40 FPS의 빠른 속도를 보여줍니다.
*   **엔드-투-엔드 학습:** 심층 컨볼루션 신경망을 사용하여 대규모 데이터셋으로 엔드-투-엔드 오프라인 학습이 가능하도록 설계되었습니다.

## 📎 Related Works
*   **Siamese Network 기반 시각 추적기:**
    *   **SiamFC**: Siamese 네트워크를 특징 추출기로 사용하고 상관관계(correlation) 계층을 도입하여 효율적인 추적을 가능하게 했습니다. 하지만 크기 변화에 대처하기 위해 다중 스케일 테스트가 필요하며 종횡비 변화는 처리하지 못했습니다.
    *   **SiamRPN**: SiamFC에 RPN(Region Proposal Network)을 도입하여 더 정확한 경계 박스를 얻었습니다.
    *   **SiamRPN++, SiamMask, SiamDW**: ResNet 등 더 깊은 네트워크를 사용하고 패딩(padding) 문제 등을 해결하여 SiamRPN의 성능을 더욱 향상시켰습니다.
    *   **제한점**: 이러한 앵커 기반(anchor-based) 추적기들은 앵커 박스의 파라미터를 신중하게 설계하고 고정해야 하며, 이는 많은 경험적 조정과 트릭을 필요로 합니다.
*   **앵커 프리(Anchor-free) 객체 탐지기:**
    *   **DenseBox, UnitBox, YOLOv1**: 초기의 앵커 프리 탐지 방식으로, FCN 또는 그리드 기반 예측을 사용했습니다.
    *   **CornerNet, ExtremeNet, RepPoints**: 키포인트 기반(keypoint-based) 탐지 방식을 제안했습니다.
    *   **FSAF, FCOS**: FCOS는 앵커 참조 없이 직접 객체 존재 가능성과 경계 박스 좌표를 예측하는 방식입니다.
    *   **추적에 대한 도전**: 객체 탐지기와 달리 추적에서는 "알 수 없는 범주(unknown categories)"의 객체를 다루며, 다른 객체들 사이에서 "동일한 객체를 식별(discrimination between different objects)"해야 하므로, 대상 외형 정보를 인코딩하는 템플릿 브랜치(template branch)가 필요합니다.

## 🛠️ Methodology
SiamBAN은 Siamese 네트워크 백본(backbone)과 여러 개의 Box Adaptive Head로 구성됩니다.

*   **Siamese 네트워크 백본:**
    *   ResNet-50을 백본 네트워크로 채택합니다.
    *   세부 공간 정보를 유지하고 수용 영역(receptive field)을 넓히기 위해 `conv4` 및 `conv5` 블록의 다운샘플링 연산을 제거하고 Atrous 컨볼루션(Atrous Convolution)을 사용합니다 (각각 Atrous rate 2, 4).
    *   템플릿 브랜치($z$)와 탐색 브랜치($x$)의 두 개의 동일한 브랜치로 구성되며, 파라미터를 공유합니다.
    *   계산 부담을 줄이기 위해 $1 \times 1$ 컨볼루션을 사용하여 특징 채널을 256으로 줄이고, 템플릿 브랜치의 중앙 $7 \times 7$ 영역만 사용합니다.
*   **Box Adaptive Head:**
    *   분류 모듈(classification module)과 회귀 모듈(regression module)로 구성됩니다.
    *   두 모듈 모두 템플릿 브랜치 특징 `$[\phi(z)]_{cls/reg}$`과 탐색 브랜치 특징 `$[\phi(x)]_{cls/reg}$`을 입력으로 받습니다.
    *   깊이별 교차 상관관계(depth-wise cross-correlation)를 사용하여 특징 맵을 결합합니다:
        $$P^{\text{w} \times \text{h} \times 2}_{\text{cls}} = [\phi(x)]_{\text{cls}} \star [\phi(z)]_{\text{cls}}$$
        $$P^{\text{w} \times \text{h} \times 4}_{\text{reg}} = [\phi(x)]_{\text{reg}} \star [\phi(z)]_{\text{reg}}$$
        여기서 $\star$는 컨볼루션 연산을 의미합니다.
    *   각 공간 위치에 대해 전경-배경 분류 점수(2채널)와 경계 박스의 네 변까지의 상대적인 오프셋(offset)을 나타내는 4D 벡터를 직접 예측합니다 ($d_l, d_t, d_r, d_b$).
    *   회귀 모듈의 마지막 계층에는 양수 값을 보장하기 위해 $\exp(x)$ 함수를 적용합니다.
*   **다중 레벨 예측(Multi-level Prediction):**
    *   `conv3`, `conv4`, `conv5` 블록의 특징 맵을 활용하여 다중 레벨 예측을 수행합니다. 각 레벨의 특징은 다른 수용 영역을 가지므로 상이한 정보를 포착합니다.
    *   각 탐지 헤드에서 얻은 분류 맵과 회귀 맵은 적응적으로 융합됩니다:
        $$P^{\text{w} \times \text{h} \times 2}_{\text{cls-all}} = \sum^{5}_{l=3} \alpha_{l} P^{l}_{\text{cls}}$$
        $$P^{\text{w} \times \text{h} \times 4}_{\text{reg-all}} = \sum^{5}_{l=3} \beta_{l} P^{l}_{\text{reg}}$$
        여기서 $\alpha_l, \beta_l$는 네트워크와 함께 최적화되는 학습 가능한 가중치입니다.
*   **손실 함수:**
    *   분류 손실($L_{\text{cls}}$)로는 교차 엔트로피(cross entropy) 손실을, 회귀 손실($L_{\text{reg}}$)로는 IoU(Intersection over Union) 손실을 사용합니다.
    *   총 손실은 $L=\lambda_1 L_{\text{cls}} + \lambda_2 L_{\text{reg}}$이며, $\lambda_1 = \lambda_2 = 1$로 설정됩니다.
*   **샘플 레이블 할당:**
    *   새로운 타원형(ellipse) 기반의 샘플 레이블 할당 방식을 제안합니다.
    *   대상 경계 박스 중앙을 기준으로 두 개의 타원 $E_1, E_2$를 정의합니다.
    *   내부 타원 $E_2$ 내부에 위치하는 점은 양성(positive) 샘플로, 외부 타원 $E_1$ 외부에 위치하는 점은 음성(negative) 샘플로 할당하고, 그 사이는 무시합니다. 이는 대상의 크기 및 종횡비를 고려하여 전경-배경을 더욱 정확하게 구별합니다.
*   **학습 및 추론:**
    *   ImageNet VID, YouTube-BoundingBoxes, COCO, GOT10k, LaSOT 등의 대규모 데이터셋으로 엔드-투-엔드 학습을 진행합니다.
    *   추론 시, 첫 프레임에서 템플릿 특징을 추출하여 캐시하고, 이후 프레임에서는 탐색 패치를 잘라 특징을 추출한 후 예측을 수행합니다. 최적의 점수를 가진 예측 박스를 선택하고 선형 보간을 통해 크기를 업데이트합니다.

## 📊 Results
SiamBAN은 여러 시각 객체 추적 벤치마크에서 최첨단 성능을 달성하며 40 FPS의 속도를 유지합니다.

*   **VOT2018:**
    *   EAO(Expected Average Overlap): 0.452로 최고 성능을 달성했습니다.
    *   정확도(Accuracy)는 0.597로 DiMP와 동률로 2위, 견고성(Robustness)은 0.178로 우수한 성능을 보였습니다.
    *   특히 폐색(occlusion), 크기 변화(size change), 움직임 변화(motion change) 속성에서 1위를 차지했습니다.
*   **VOT2019 (실시간 실험):**
    *   EAO 0.327, 정확도 0.602로 모두 최고 성능을 달성했습니다.
    *   SiamRPN++ 대비 EAO에서 14.7% 상대적 이득, 실패율에서 17.8% 감소를 보여주며, 대상 경계 박스 추정의 정확성을 입증했습니다.
*   **OTB100:**
    *   성공도(Success) 플롯 AUC에서 0.696으로 SiamRPN++와 동률로 최고 성능을 기록했습니다.
*   **NFS (30FPS):** AUC 0.594로 2위를 차지했습니다.
*   **UAV123:** 성공도 플롯 AUC 0.631로 최첨단 추적기들과 경쟁력 있는 성능을 보였습니다.
*   **LaSOT:** 성공도 플롯 AUC 0.514로 3위, 정규화된 정확도(Normalized Precision)에서 0.598로 2위를 차지했으며, SiamRPN++보다 5.1% 높은 성능을 보였습니다.

## 🧠 Insights & Discussion
*   **다중 레벨 예측의 중요성:** Ablation Study를 통해 `conv3`, `conv4`, `conv5`와 같은 다중 레벨 특징을 통합하여 사용하는 것이 성능 향상에 결정적임을 확인했습니다. 초기 레이어는 정밀한 지역화에 유용한 미세한 정보를 포착하고, 후기 레이어는 대상 외형 변화에 강인한 추상적인 의미 정보를 인코딩하므로, 이들의 조합이 최적의 결과를 이끌어냅니다.
*   **새로운 샘플 레이블 할당 방식의 효과:** 제안된 타원형 기반 레이블 할당 방식이 원형(circle) 또는 사각형(rectangle) 기반 방식보다 우수한 성능을 보였습니다. 이는 타원형 레이블이 대상의 크기와 종횡비를 더 정확하게 반영하여 전경-배경 샘플을 더 잘 구분함으로써, 학습된 추적기의 견고성을 높이기 때문입니다. 이러한 통찰은 샘플 레이블 할당이 추적기 성능에 미치는 중요성을 강조합니다.
*   **온라인 업데이트 없는 고성능:** SiamBAN은 온라인 업데이트 없이도 DiMP와 같은 온라인 업데이트 방식의 추적기에 버금가는 또는 그 이상의 성능을 달성하며, 이는 모델의 강력한 오프라인 학습 능력과 일반화 능력을 시사합니다.
*   **간결한 설계와 효율성:** 앵커 박스나 다중 스케일 탐색과 같은 복잡한 요소 없이 FCN 내에서 직접 분류 및 회귀를 수행하는 간결한 설계가 고성능과 빠른 속도를 동시에 달성할 수 있음을 보여줍니다.

## 📌 TL;DR
SiamBAN은 시각 객체 추적에서 대상의 크기 및 종횡비 추정의 비효율성을 해결하기 위해 제안된 앵커 프리(anchor-free) 추적 프레임워크입니다. 이 방법은 완전 컨볼루션 네트워크(FCN)를 사용하여 추적 문제를 분류 및 회귀 문제로 전환하고, 사전 정의된 앵커 박스나 다중 스케일 탐색 없이 직접 경계 박스를 예측합니다. 백본으로 수정된 ResNet-50과 다중 레벨 특징 융합, 그리고 타원형 기반 샘플 레이블 할당을 통해 높은 정확도와 견고성을 확보합니다. 실험 결과, SiamBAN은 여러 벤치마크에서 최첨단 성능을 달성하며 40 FPS의 실시간 속도를 제공하여 효율성과 효과성을 모두 입증했습니다.