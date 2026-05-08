# CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools

Chinedu Innocent Nwoye, Kareem Elgohary, Anvita Srinivas, Fauzan Zaid, Joël L. Lavanchy, Nicolas Padoy (2025)

## 🧩 Problem to Solve

수술 비디오에서의 도구 추적(Tool Tracking)은 수술 기술 평가, 안전 구역 추정, 인간-기계 협업과 같은 컴퓨터 보조 중재(computer-assisted interventions)를 발전시키는 데 필수적이다. 그러나 기존의 데이터셋들은 지나치게 일반적인 추적 형식(tracking formalizations)에 의존하고 있어, 도구가 카메라의 시야(Field of View, FoV)를 벗어나거나 신체 밖으로 나가는 것과 같은 수술 특유의 역동성을 포착하지 못한다는 한계가 있다.

이러한 데이터의 부족은 임상적으로 유의미한 궤적(trajectory) 분석을 어렵게 만들며, 특히 연기(smoke), 반사(reflection), 출혈(bleeding)과 같은 시각적 챌린지 상황에서 AI 모델의 성능을 저하시킨다. 따라서 본 논문의 목표는 수술 도메인의 특수성을 반영하여 다각적 관점에서의 추적을 가능하게 하는 전문화된 데이터셋인 CholecTrack20을 구축하고, 이를 통해 현재의 추적 알고리즘 수준을 벤치마킹하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 수술 도구 추적을 위한 새로운 관점인 **Multi-Perspective (MP) Tracking** 개념을 도입하고, 이를 구현한 **CholecTrack20** 데이터셋을 제안한 것이다.

단순히 화면에 보이는 도구를 추적하는 것을 넘어, 도구의 생애 주기와 물리적 위치를 기준으로 세 가지 관점(Intraoperative, Intracorporeal, Visibility)에서 궤적을 정의하였다. 이를 통해 임상적 맥락에 따라 유연하게 적용 가능한 도구 궤적 데이터를 제공하며, 도구의 카테고리, 조작자(operator), 수술 단계(phase), 시각적 챌린지 등 매우 상세한 어노테이션을 포함하여 AI 모델이 수술 환경의 복잡성을 학습할 수 있도록 설계하였다.

## 📎 Related Works

기존의 객체 검출 및 추적 연구는 주로 COCO, KITTI, MOTChallenge와 같은 일반 데이터셋을 기반으로 발전해 왔으며, 단일 객체 추적(SOT), 다중 객체 추적(MOT), 다중 클래스 다중 객체 추적(MCMOT)으로 구분된다. 하지만 이러한 일반적인 접근 방식은 출혈, 연기, 급격한 움직임 및 가변적인 조명과 같은 수술실 특유의 환경에서 성능이 크게 저하된다.

수술 도구 추적 분야에서도 이미지 기반 접근 방식이 시도되었으나, 도구가 교체되거나 시야를 벗어났다 재진입할 때 발생하는 ID 단절(identity fragmentation) 및 ID 전환(identity switch) 문제로 인해 정확도가 낮았다. 기존의 수술 데이터셋들 역시 추적 관점이 단일하여 수술 도구의 복잡한 상호작용과 사용 패턴을 포괄적으로 모델링하는 데 한계가 있었다.

## 🛠️ Methodology

### 전체 파이프라인 및 데이터 수집

CholecTrack20은 복강경 담낭 절제술(laparoscopic cholecystectomy) 비디오를 기반으로 한다. Cholec80 및 CholecT50 데이터셋에서 추출한 20개의 전체 길이 수술 비디오를 사용하였으며, 어노테이션의 효율성과 일관성을 위해 25 FPS의 원본 영상을 1 FPS로 샘플링하여 총 35,000 프레임 이상을 확보하였다.

### Track Formalization

도구 추적은 시간 $t$에서 프레임 $t$의 $i$번째 도구와 프레임 $t+1$의 $j$번째 도구를 연결하는 연관 행렬(association matrix) $A(t)$를 해결하는 문제로 정의된다.
$$A_{i,j} = \begin{cases} 1, & \text{if tool } i \text{ at } t \text{ is associated with tool } j \text{ at } t+1 \\ 0, & \text{otherwise} \end{cases}$$
이 과정에서 단순한 시각적 특징뿐만 아니라, 도구가 삽입된 트로카 포트(trocar port)의 위치와 외과의의 손 위치 등 임상적 지식을 함께 고려하여 고유 ID를 부여한다.

### Multi-Perspective (MP) Trajectory

본 논문은 추적의 시작과 끝을 정의하는 세 가지 관점을 제안한다.

1. **Intraoperative trajectory (수술 내 궤적):** 도구가 환자의 몸에 처음 나타난 순간부터 마지막으로 사라질 때까지의 전체 생애 주기를 추적한다. 시야 밖으로 나갔다 돌아오는 경우 Re-identification(re-ID)이 필요하며, 도구 사용량 모니터링 및 숙련도 평가에 사용된다.
2. **Intracorporeal trajectory (체내 궤적):** 도구가 신체 내부로 진입한 시점부터 트로카 포트를 통해 완전히 나가는 시점까지를 추적한다. 카메라 시야 밖에서 일어나는 동작(예: 조직을 잡아 고정하는 동작)까지 포함하며, 수술 워크플로우 이해와 위험 예측에 중요하다.
3. **Visibility trajectory (가시성 궤적):** 도구가 카메라 시야 내에 처음 나타난 순간부터 사라질 때까지를 추적한다. 짧은 가려짐(occlusion)은 2초의 허용 오차 내에서 동일 ID로 유지하며, 실시간 피드백 및 동작 경제성 측정에 활용된다.

### 어노테이션 세부 사항

- **도구 카테고리 (7종):** Cold grasper, Bipolar grasper, Monopolar hook, Monopolar scissors, Clipper, Irrigator, Specimen bag.
- **조작자 (4종):** 주 수술자 왼손(MSLH), 주 수술자 오른손(MSRH), 보조 수술자 오른손(ASRH), NULL.
- **시각적 챌린지 (8종):** Blurring, Bleeding, Lens fouling, Crowded scene, Occlusion, Smoke, Reflection, Trocar view.
- **수술 단계 (7단계):** Preparation, Calot triangle dissection, Gallbladder dissection, Clipping & cutting, Gallbladder packaging, Cleaning and coagulation, Gallbladder extraction.

## 📊 Results

### 도구 검출(Tool Detection) 벤치마크

다양한 SOTA 검출기들을 평가한 결과, **YOLOv7**이 $AP_{0.5:0.95} = 56.1\%$로 가장 높은 성능을 보였다.

- **특징:** Hook은 모든 모델에서 가장 높은 검출률을 보인 반면, Irrigator와 Specimen bag은 경계가 불분명하거나 변형이 심해 검출이 어려웠다.
- **챌린지별:** 출혈(bleeding)과 연기(smoke) 상황에서는 비교적 강건했으나, 흐림(blurring)이나 반사(reflection) 상황에서는 모든 모델의 성능이 저하되었다.

### 도구 추적(Tool Tracking) 벤치마크

Bot-SORT, ByteTrack, SMILETrack 등 최신 MOT 알고리즘을 적용하여 분석하였다.

- **관점별 성능:**
  - **Visibility $\rightarrow$ Intracorporeal $\rightarrow$ Intraoperative** 순으로 난이도가 낮았다.
  - Visibility 추적에서는 Bot-SORT가 HOTA 44.7%로 가장 높았으나, Intracorporeal 추적에서는 27.0%로 크게 하락하였다. 이는 체내 진입/퇴출 시점이 시각적으로 명확하지 않기 때문이다.
- **도구별 성능:** Grasper가 가장 높은 추적 정확도를 보였으며, Specimen bag은 형태 변형으로 인해 추적이 가장 어려웠다.
- **수술 단계별:** Clipping & cutting 단계가 동작이 단순하고 선형적이어서 가장 추적하기 쉬웠으며, 담낭 박리(dissection) 단계는 도구 조작이 빈번하여 더 어려웠다.

## 🧠 Insights & Discussion

본 연구의 실험 결과, 현재의 SOTA 추적 모델들은 **HOTA 45% 미만**의 성능을 보이며, 이는 실제 임상 환경에 적용하기에는 턱없이 부족한 수준이다.

가장 큰 문제는 현재의 모델들이 주로 **외형(appearance)과 위치(location)** 정보에만 의존한다는 점이다. 수술 도구들은 서로 외형이 매우 유사하며, 시야 밖으로 나갔다 돌아오는 경우가 빈번하기 때문에 단순한 시각적 특징만으로는 ID를 유지하기 어렵다. 따라서 향후 연구는 단순한 픽셀 정보 이상의 **문맥 인식(context-aware)** Re-ID 방법론, 즉 수술 단계나 조작자의 특성을 결합한 추적 알고리즘으로 발전해야 한다.

또한, Intracorporeal trajectory의 낮은 성능은 AI가 카메라에 보이지 않는 영역의 상태를 추론해야 함을 시사하며, 이는 수술 도구의 이력(history)과 물리적 제약 조건을 학습하는 모델의 필요성을 뒷받침한다.

## 📌 TL;DR

본 논문은 수술 도구 추적의 임상적 유의성을 높이기 위해 **Intraoperative, Intracorporeal, Visibility**라는 세 가지 관점의 궤적 정의를 도입한 **CholecTrack20** 데이터셋을 제안한다. 20개의 수술 영상과 65K개의 라벨을 통해 벤치마킹을 수행한 결과, 현재의 SOTA 모델들은 시각적 챌린지와 도구의 유사성 문제로 인해 임상 적용 수준의 정확도를 확보하지 못했음을 확인하였다. 이 데이터셋은 향후 문맥 인식 기반의 강건한 수술 보조 AI 시스템 개발을 위한 핵심 기초 자산이 될 것으로 기대된다.
