#!/usr/bin/env python3
"""Generate and validate a wholly SYNTHETIC academic dataset for 59,000 students.

Requirements: Python >=3.10, pandas >=2.0, numpy >=1.24.
No network, credentials, real student identifiers or original PDF are required.
Usage: python generate_dataset.py --output-dir UQU_59000_Student_Timetables
       python generate_dataset.py --output-dir PATH --validate-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261026
N_STUDENTS = 59000
DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
DAY_ORDER = dict(zip(DAYS, range(5)))
SCHEMAS = {
    'students': 'Student_ID,College,Department,Major,Academic_Level,Campus,Gender'.split(','),
    'courses': 'Course_Code,Course_Name_AR,Course_Name_EN,College,Department,Credit_Hours,Course_Level'.split(','),
    'sections': 'Section_ID,Course_Code,Section_Number,Activity_Type,Instructor_ID,Capacity,Campus,Building,Room,Delivery_Mode'.split(','),
    'section_meetings': 'Meeting_ID,Section_ID,Day,Start_Time,End_Time,Campus,Building,Room,Delivery_Mode'.split(','),
    'student_enrollments': 'Student_ID,Section_ID,Course_Code'.split(','),
    'student_timetables': 'Student_ID,College,Major,Academic_Level,Day,Course_Code,Course_Name,Section_ID,Section_Number,Activity_Type,Start_Time,End_Time,Campus,Building,Room,Delivery_Mode'.split(','),
    'validation_report': 'Validation_Check,Result,Error_Count,Details'.split(','),
}

# Every allocation below is an explicit scenario assumption, NOT UQU statistics.
# college: display name, building, lab propensity, (major, AR name, seats, female share).
COLLEGES = {
 'BUS': ('كلية إدارة الأعمال', 30, .10, [('BUS','إدارة الأعمال',4000,.51),('ACC','المحاسبة',3500,.50),('MKT','التسويق',3000,.56)]),
 'SHA': ('كلية الشريعة والدراسات الإسلامية', 31, .00, [('SHR','الشريعة',5000,.48),('ISL','الدراسات الإسلامية',3000,.56)]),
 'EDU': ('كلية التربية', 32, .10, [('EDU','التربية',4000,.62),('ECE','الطفولة المبكرة',3500,.80)]),
 'SOC': ('كلية العلوم الاجتماعية', 33, .15, [('PSY','علم النفس',2500,.62),('SOC','علم الاجتماع',2000,.60),('GEO','الجغرافيا',2000,.47)]),
 'CMP': ('كلية الحاسبات', 12, .48, [('AI','الذكاء الاصطناعي',2200,.43),('CS','علوم الحاسب',2000,.46),('IS','نظم المعلومات',1800,.55)]),
 'SCI': ('كلية العلوم', 34, .58, [('MAT','الرياضيات',2000,.55),('PHY','الفيزياء',1700,.50),('CHE','الكيمياء',1800,.57)]),
 'ENG': ('كلية الهندسة', 35, .62, [('CIV','الهندسة المدنية',1800,.24),('ELE','الهندسة الكهربائية',1500,.27),('MEC','الهندسة الميكانيكية',1200,.22)]),
 'AHS': ('كلية العلوم الطبية التطبيقية', 36, .64, [('NUR','التمريض',1600,.66),('MLS','علوم المختبرات الطبية',1300,.58),('PHT','العلاج الطبيعي',1100,.55)]),
 'MED': ('كلية الطب', 37, .62, [('MED','الطب والجراحة',2500,.50)]),
 'ARB': ('كلية اللغة العربية', 38, .00, [('ARB','اللغة العربية',2200,.58)]),
 'PHR': ('كلية الصيدلة', 39, .56, [('PHR','الصيدلة',1800,.57)]),
}


def pairs(ar: str, en: str) -> list[tuple[str, str]]:
    a, b = ar.split('|'), en.split('|')
    assert len(a) == len(b), (len(a), len(b), ar)
    return list(zip(a, b))


# Two shared college courses at each level. Names/codes except PDF anchors are synthetic.
SHARED = {
'BUS': pairs('مبادئ الإدارة|رياضيات الأعمال|مبادئ الاقتصاد الجزئي|إحصاء الأعمال|مبادئ الاقتصاد الكلي|السلوك التنظيمي|قانون الأعمال|الاقتصاد الإداري|مناهج البحث في الأعمال|ريادة الأعمال|إدارة العمليات|حوكمة الشركات|استراتيجية الأعمال|أخلاقيات الأعمال|إدارة المشروعات|دراسة حالة في الأعمال', 'Principles of Management|Business Mathematics|Microeconomics|Business Statistics|Macroeconomics|Organizational Behavior|Business Law|Managerial Economics|Business Research Methods|Entrepreneurship|Operations Management|Corporate Governance|Business Strategy|Business Ethics|Project Management|Business Case Study'),
'SHA': pairs('مدخل إلى الدراسات الشرعية|اللغة العربية للدراسات الشرعية|تاريخ التشريع|مصطلح الحديث|مناهج البحث الشرعي|قواعد الاستدلال|مقاصد الشريعة|أصول التفسير|القواعد الفقهية|فقه الأسرة|فقه المعاملات|السياسة الشرعية|النوازل المعاصرة|الاقتصاد الإسلامي|حلقة بحث شرعي|مهارات الفتوى', 'Introduction to Sharia Studies|Arabic for Sharia Studies|History of Islamic Legislation|Hadith Terminology|Sharia Research Methods|Principles of Evidence|Objectives of Sharia|Principles of Quranic Exegesis|Legal Maxims|Family Jurisprudence|Transactions Jurisprudence|Islamic Governance|Contemporary Legal Issues|Islamic Economics|Sharia Research Seminar|Fatwa Skills'),
'EDU': pairs('مدخل إلى التربية|علم النفس التربوي|نظريات التعلم|تاريخ التربية|المناهج الدراسية|تقنيات التعليم|القياس والتقويم|إدارة الصف|تصميم التدريس|التربية الدامجة|البحث التربوي|التوجيه والإرشاد|التدريس المصغر|تقويم المناهج|الممارسة التربوية|حلقة بحث تربوي', 'Introduction to Education|Educational Psychology|Learning Theories|History of Education|Curriculum Studies|Educational Technology|Measurement and Assessment|Classroom Management|Instructional Design|Inclusive Education|Educational Research|Guidance and Counseling|Microteaching|Curriculum Evaluation|Educational Practice|Education Research Seminar'),
'SOC': pairs('مدخل إلى العلوم الاجتماعية|التفكير النقدي|إحصاء اجتماعي|تاريخ الفكر الاجتماعي|مناهج البحث الاجتماعي|مهارات المقابلة|تحليل البيانات الاجتماعية|المجتمع السعودي|أخلاقيات البحث|السياسات الاجتماعية|المسح الميداني|المجتمع والبيئة|التنمية الاجتماعية|البحث النوعي|تطبيقات ميدانية|حلقة بحث اجتماعي', 'Introduction to Social Sciences|Critical Thinking|Social Statistics|History of Social Thought|Social Research Methods|Interviewing Skills|Social Data Analysis|Saudi Society|Research Ethics|Social Policy|Field Survey Methods|Society and Environment|Social Development|Qualitative Research|Field Applications|Social Research Seminar'),
'CMP': pairs('مقدمة في الحوسبة|رياضيات متقطعة|برمجة الحاسب 1|التفاضل والتكامل|برمجة الحاسب 2|الجبر الخطي|هياكل البيانات|الاحتمالات والإحصاء|قواعد البيانات|هندسة البرمجيات|شبكات الحاسب|نظم التشغيل|أخلاقيات الحوسبة|مناهج البحث التقني|تطوير المشروعات التقنية|ضمان جودة البرمجيات', 'Introduction to Computing|Discrete Mathematics|Computer Programming 1|Calculus|Computer Programming 2|Linear Algebra|Data Structures|Probability and Statistics|Database Systems|Software Engineering|Computer Networks|Operating Systems|Computing Ethics|Technical Research Methods|Technical Project Development|Software Quality Assurance'),
'SCI': pairs('التفاضل والتكامل 1|فيزياء عامة 1|التفاضل والتكامل 2|كيمياء عامة|الجبر الخطي|إحصاء علمي|المعادلات التفاضلية|الطرق العددية|الكتابة العلمية|تصميم التجارب|مناهج البحث العلمي|الحوسبة العلمية|السلامة المختبرية|نمذجة علمية|حلقة بحث علمي|تطبيقات علمية متقدمة', 'Calculus 1|General Physics 1|Calculus 2|General Chemistry|Linear Algebra|Scientific Statistics|Differential Equations|Numerical Methods|Scientific Writing|Experimental Design|Scientific Research Methods|Scientific Computing|Laboratory Safety|Scientific Modeling|Scientific Research Seminar|Advanced Scientific Applications'),
'ENG': pairs('التفاضل والتكامل الهندسي|الرسم الهندسي|فيزياء هندسية|برمجة للمهندسين|ميكانيكا هندسية|مواد هندسية|المعادلات التفاضلية الهندسية|احتمالات هندسية|الاقتصاد الهندسي|القياسات الهندسية|أخلاقيات الهندسة|إدارة المشروعات الهندسية|التصميم الهندسي|السلامة الهندسية|مشروع تصميم 1|مشروع تصميم 2', 'Engineering Calculus|Engineering Drawing|Engineering Physics|Programming for Engineers|Engineering Mechanics|Engineering Materials|Engineering Differential Equations|Engineering Probability|Engineering Economics|Engineering Measurements|Engineering Ethics|Engineering Project Management|Engineering Design|Engineering Safety|Design Project 1|Design Project 2'),
'AHS': pairs('أحياء طبية|كيمياء حيوية|تشريح عام|وظائف الأعضاء|أحياء دقيقة طبية|إحصاء حيوي|علم الأمراض|مكافحة العدوى|أخلاقيات المهن الصحية|علم الأدوية|البحث الصحي|التثقيف الصحي|الممارسة المبنية على الدليل|مهارات سريرية|مشروع بحث صحي|حلقة تطبيقات صحية', 'Medical Biology|Biochemistry|General Anatomy|Physiology|Medical Microbiology|Biostatistics|Pathology|Infection Control|Health Professions Ethics|Pharmacology|Health Research|Health Education|Evidence Based Practice|Clinical Skills|Health Research Project|Health Applications Seminar'),
'MED': pairs('أساسيات الطب|الكيمياء الطبية|التشريح 1|وظائف الأعضاء 1|التشريح 2|وظائف الأعضاء 2|الكيمياء الحيوية الطبية|علم الأنسجة|علم الأمراض 1|أحياء دقيقة سريرية|علم الأمراض 2|علم المناعة|مبادئ الطب السريري|الإحصاء الطبي|مهارات التواصل الطبي|علم الوبائيات|البحث الطبي|الصحة العامة|أخلاقيات الطب|الممارسة المبنية على الدليل|مراجعة الحالات السريرية|طب المجتمع|السلامة السريرية|حلقة بحث طبي', 'Foundations of Medicine|Medical Chemistry|Anatomy 1|Physiology 1|Anatomy 2|Physiology 2|Medical Biochemistry|Histology|Pathology 1|Clinical Microbiology|Pathology 2|Immunology|Principles of Clinical Medicine|Medical Statistics|Medical Communication|Epidemiology|Medical Research|Public Health|Medical Ethics|Evidence Based Medicine|Clinical Case Review|Community Medicine|Clinical Safety|Medical Research Seminar'),
'ARB': pairs('مدخل إلى علوم العربية|المهارات اللغوية|النحو 1|الصرف 1|النحو 2|الصرف 2|البلاغة 1|علم الأصوات|البلاغة 2|تاريخ الأدب|العروض والقافية|مناهج البحث اللغوي|النقد الأدبي|تحقيق النصوص|حلقة بحث لغوي|الكتابة المتخصصة', 'Introduction to Arabic Studies|Language Skills|Grammar 1|Morphology 1|Grammar 2|Morphology 2|Rhetoric 1|Phonetics|Rhetoric 2|History of Literature|Prosody|Linguistic Research Methods|Literary Criticism|Textual Editing|Linguistics Research Seminar|Specialized Writing'),
'PHR': pairs('أحياء للصيدلة|كيمياء عامة للصيدلة|تشريح للصيدلة|وظائف الأعضاء للصيدلة|كيمياء عضوية صيدلية|كيمياء حيوية صيدلية|أحياء دقيقة صيدلية|إحصاء صيدلي|علم الأمراض للصيدلة|علم المناعة للصيدلة|البحث الصيدلي|أخلاقيات الصيدلة|الصحة العامة للصيدلة|اقتصاديات الدواء|الممارسة المبنية على الدليل|سلامة الدواء|الرعاية الصحية المتكاملة|الكتابة الصيدلية|تقييم الدراسات الدوائية|حلقة بحث صيدلي', 'Biology for Pharmacy|General Chemistry for Pharmacy|Anatomy for Pharmacy|Physiology for Pharmacy|Pharmaceutical Organic Chemistry|Pharmaceutical Biochemistry|Pharmaceutical Microbiology|Pharmacy Statistics|Pathology for Pharmacy|Immunology for Pharmacy|Pharmacy Research|Pharmacy Ethics|Public Health for Pharmacy|Pharmacoeconomics|Evidence Based Pharmacy|Medication Safety|Integrated Healthcare|Pharmaceutical Writing|Drug Study Evaluation|Pharmacy Research Seminar'),
}

# Two specialty courses at each academic level.
SPECIAL = {
'AI': pairs('مقدمة في الذكاء الاصطناعي|حل المشكلات البرمجية|أساسيات علم البيانات|البرمجة الكائنية|البحث في فضاءات الحالة|تمثيل المعرفة|التعلم الإحصائي|تحليل الخوارزميات|مبادىء وتقنيات الذكاء الاصطناعي|تعلم الآلة|التعلم العميق|معالجة اللغة الطبيعية|الرؤية الحاسوبية|التعلم المعزز|تطبيقات الذكاء الاصطناعي|مشروع ذكاء اصطناعي', 'Introduction to Artificial Intelligence|Programming Problem Solving|Foundations of Data Science|Object Oriented Programming|State Space Search|Knowledge Representation|Statistical Learning|Algorithm Analysis|Principles and Techniques of AI|Machine Learning|Deep Learning|Natural Language Processing|Computer Vision|Reinforcement Learning|Applied Artificial Intelligence|AI Project'),
'CS': pairs('مبادئ البرمجة|مهارات الحاسب|برمجة تطبيقية|مقدمة في نظم الحاسب|تنظيم الحاسب|البرمجة الكائنية|تحليل الخوارزميات|برمجة النظم|نظرية الحوسبة|تصميم المترجمات|نظم موزعة|أمن البرمجيات|الحوسبة السحابية|برمجة متوازية|تطوير تطبيقات متقدم|مشروع علوم حاسب', 'Programming Principles|Computer Skills|Applied Programming|Introduction to Computer Systems|Computer Organization|Object Oriented Programming|Algorithm Analysis|Systems Programming|Theory of Computation|Compiler Design|Distributed Systems|Software Security|Cloud Computing|Parallel Programming|Advanced Application Development|Computer Science Project'),
'IS': pairs('مقدمة في نظم المعلومات|مهارات التحليل|برمجة الأعمال|معلومات الأعمال|تحليل النظم|نمذجة البيانات|تصميم النظم|نظم المؤسسات|ذكاء الأعمال|إدارة المعرفة|إدارة خدمات التقنية|تحليل الأعمال|التجارة الإلكترونية|حوكمة تقنية المعلومات|نظم دعم القرار|مشروع نظم معلومات', 'Introduction to Information Systems|Analytical Skills|Business Programming|Business Information|Systems Analysis|Data Modeling|Systems Design|Enterprise Systems|Business Intelligence|Knowledge Management|IT Service Management|Business Analysis|Electronic Commerce|IT Governance|Decision Support Systems|Information Systems Project'),
'BUS': pairs('مقدمة في الأعمال|مهارات إدارية|الاتصال الإداري|إدارة الموارد البشرية|الإدارة المالية|إدارة التسويق|إدارة الجودة|نظم المعلومات الإدارية|التخطيط الإداري|إدارة سلاسل الإمداد|إدارة التغيير|الإدارة الدولية|القيادة الإدارية|إدارة المخاطر|إدارة الأعمال الرقمية|مشروع إدارة أعمال', 'Introduction to Business|Management Skills|Managerial Communication|Human Resource Management|Financial Management|Marketing Management|Quality Management|Management Information Systems|Management Planning|Supply Chain Management|Change Management|International Management|Leadership|Risk Management|Digital Business Management|Business Management Project'),
'ACC': pairs('مبادئ المحاسبة 1|مهارات محاسبية|مبادئ المحاسبة 2|التطبيقات المحاسبية|محاسبة متوسطة 1|محاسبة التكاليف|محاسبة متوسطة 2|نظم المعلومات المحاسبية|المحاسبة الإدارية|المراجعة 1|المحاسبة الضريبية|المراجعة 2|المحاسبة الحكومية|التحليل المالي|معايير التقارير المالية|مشروع محاسبة', 'Accounting Principles 1|Accounting Skills|Accounting Principles 2|Accounting Applications|Intermediate Accounting 1|Cost Accounting|Intermediate Accounting 2|Accounting Information Systems|Managerial Accounting|Auditing 1|Tax Accounting|Auditing 2|Government Accounting|Financial Analysis|Financial Reporting Standards|Accounting Project'),
'MKT': pairs('مقدمة في التسويق|الاتصال التسويقي|سلوك المستهلك|بحوث التسويق|إدارة المنتجات|إدارة المبيعات|التسويق الرقمي|تسويق الخدمات|استراتيجية التسعير|قنوات التوزيع|إدارة العلامة التجارية|تحليلات التسويق|التسويق الدولي|التسويق الاجتماعي|حملات تسويقية|مشروع تسويق', 'Introduction to Marketing|Marketing Communication|Consumer Behavior|Marketing Research|Product Management|Sales Management|Digital Marketing|Services Marketing|Pricing Strategy|Distribution Channels|Brand Management|Marketing Analytics|International Marketing|Social Marketing|Marketing Campaigns|Marketing Project'),
'SHR': pairs('فقه العبادات 1|أصول الفقه 1|فقه العبادات 2|أصول الفقه 2|فقه المعاملات 1|آيات الأحكام|فقه المعاملات 2|أحاديث الأحكام|فقه الجنايات|الفرائض|فقه القضاء|فقه الوقف|الفقه المقارن|العقود المالية|التطبيق القضائي|بحث تخرج في الشريعة', 'Worship Jurisprudence 1|Principles of Jurisprudence 1|Worship Jurisprudence 2|Principles of Jurisprudence 2|Transactions Jurisprudence 1|Legal Verses|Transactions Jurisprudence 2|Legal Hadith|Criminal Jurisprudence|Inheritance Law|Judicial Jurisprudence|Endowment Law|Comparative Jurisprudence|Financial Contracts|Judicial Applications|Sharia Graduation Research'),
'ISL': pairs('علوم القرآن|علوم الحديث|التفسير التحليلي 1|السيرة النبوية|التفسير التحليلي 2|العقيدة الإسلامية|الحديث التحليلي 1|تاريخ الدعوة|الحديث التحليلي 2|مناهج المفسرين|الفرق والمذاهب|أصول الدعوة|مقارنة الأديان|فقه الدعوة|الدراسات الحديثية|بحث دراسات إسلامية', 'Quranic Sciences|Hadith Sciences|Analytical Exegesis 1|Prophetic Biography|Analytical Exegesis 2|Islamic Creed|Analytical Hadith 1|History of Islamic Outreach|Analytical Hadith 2|Methods of Exegesis|Schools of Thought|Principles of Outreach|Comparative Religion|Outreach Jurisprudence|Hadith Studies|Islamic Studies Research'),
'EDU': pairs('مهارات المعلم|التربية والمجتمع|طرق التدريس العامة|النمو الإنساني|التعلم النشط|تخطيط الدروس|صعوبات التعلم|استراتيجيات التقويم|التعليم الإلكتروني|الإدارة المدرسية|التربية المقارنة|الموهبة والإبداع|التربية العملية 1|الإشراف التربوي|التربية العملية 2|مشروع تربوي', 'Teacher Skills|Education and Society|General Teaching Methods|Human Development|Active Learning|Lesson Planning|Learning Difficulties|Assessment Strategies|Electronic Learning|School Administration|Comparative Education|Giftedness and Creativity|Teaching Practice 1|Educational Supervision|Teaching Practice 2|Education Project'),
'ECE': pairs('مدخل إلى الطفولة المبكرة|نمو الطفل|لعب الطفل|صحة الطفل|أدب الطفل|فن الطفل|رياضيات الطفولة|لغة الطفل|مناهج رياض الأطفال|تقويم الطفل|بيئة التعلم المبكر|شراكة الأسرة|التدخل المبكر|إدارة الروضة|تطبيقات الطفولة|مشروع طفولة مبكرة', 'Introduction to Early Childhood|Child Development|Child Play|Child Health|Children Literature|Child Art|Early Mathematics|Child Language|Kindergarten Curriculum|Child Assessment|Early Learning Environment|Family Partnership|Early Intervention|Kindergarten Management|Early Childhood Applications|Early Childhood Project'),
'PSY': pairs('مقدمة في علم النفس|النمو النفسي|علم النفس المعرفي|علم النفس الاجتماعي|علم نفس الشخصية|علم النفس التجريبي|الصحة النفسية|الاختبارات النفسية|علم النفس الإكلينيكي|الإرشاد النفسي|علم النفس التنظيمي|تعديل السلوك|علم النفس العصبي|علم النفس الأسري|تطبيقات نفسية|مشروع علم نفس', 'Introduction to Psychology|Psychological Development|Cognitive Psychology|Social Psychology|Personality Psychology|Experimental Psychology|Mental Health|Psychological Testing|Clinical Psychology|Psychological Counseling|Organizational Psychology|Behavior Modification|Neuropsychology|Family Psychology|Psychology Applications|Psychology Project'),
'SOC': pairs('مقدمة في علم الاجتماع|البناء الاجتماعي|علم الاجتماع الحضري|علم الاجتماع الريفي|الأسرة والمجتمع|المشكلات الاجتماعية|علم اجتماع التنظيم|النظرية الاجتماعية|السكان والمجتمع|الخدمة الاجتماعية|علم اجتماع المعرفة|الجريمة والمجتمع|التغير الاجتماعي|المجتمع الرقمي|تطبيقات اجتماعية|مشروع علم اجتماع', 'Introduction to Sociology|Social Structure|Urban Sociology|Rural Sociology|Family and Society|Social Problems|Sociology of Organizations|Social Theory|Population and Society|Social Work|Sociology of Knowledge|Crime and Society|Social Change|Digital Society|Social Applications|Sociology Project'),
'GEO': pairs('الجغرافيا الطبيعية|الجغرافيا البشرية|مبادئ الخرائط|المناخ|جغرافية السكان|جغرافية العمران|نظم المعلومات الجغرافية 1|الاستشعار عن بعد|نظم المعلومات الجغرافية 2|الجغرافيا الاقتصادية|التخطيط المكاني|جغرافية النقل|التحليل المكاني|إدارة الموارد|تطبيقات خرائطية|مشروع جغرافيا', 'Physical Geography|Human Geography|Cartography|Climatology|Population Geography|Settlement Geography|GIS 1|Remote Sensing|GIS 2|Economic Geography|Spatial Planning|Transport Geography|Spatial Analysis|Resource Management|Cartographic Applications|Geography Project'),
'MAT': pairs('مبادئ الرياضيات|هندسة تحليلية|نظرية المجموعات|مقدمة في البرهان|جبر مجرد 1|تحليل حقيقي 1|جبر مجرد 2|تحليل حقيقي 2|تحليل مركب|نظرية الأعداد|الطوبولوجيا|التحليل العددي|التحسين الرياضي|الاحتمالات المتقدمة|ندوة رياضيات|مشروع رياضيات', 'Mathematical Foundations|Analytic Geometry|Set Theory|Introduction to Proof|Abstract Algebra 1|Real Analysis 1|Abstract Algebra 2|Real Analysis 2|Complex Analysis|Number Theory|Topology|Numerical Analysis|Mathematical Optimization|Advanced Probability|Mathematics Seminar|Mathematics Project'),
'PHY': pairs('ميكانيكا عامة|بصريات عامة|كهرباء ومغناطيسية|ديناميكا حرارية|فيزياء حديثة|إلكترونيات فيزيائية|ميكانيكا كلاسيكية|طرق رياضية للفيزياء|ميكانيكا الكم 1|فيزياء إحصائية|ميكانيكا الكم 2|فيزياء نووية|فيزياء الحالة الصلبة|فيزياء الليزر|تطبيقات فيزيائية|مشروع فيزياء', 'General Mechanics|General Optics|Electricity and Magnetism|Thermodynamics|Modern Physics|Physical Electronics|Classical Mechanics|Mathematical Physics|Quantum Mechanics 1|Statistical Physics|Quantum Mechanics 2|Nuclear Physics|Solid State Physics|Laser Physics|Physics Applications|Physics Project'),
'CHE': pairs('مبادئ الكيمياء|حسابات كيميائية|كيمياء عضوية 1|كيمياء تحليلية 1|كيمياء عضوية 2|كيمياء تحليلية 2|كيمياء غير عضوية 1|كيمياء فيزيائية 1|كيمياء غير عضوية 2|كيمياء فيزيائية 2|التحليل الآلي|كيمياء البوليمرات|الكيمياء البيئية|الكيمياء الصناعية|تطبيقات كيميائية|مشروع كيمياء', 'Chemistry Principles|Chemical Calculations|Organic Chemistry 1|Analytical Chemistry 1|Organic Chemistry 2|Analytical Chemistry 2|Inorganic Chemistry 1|Physical Chemistry 1|Inorganic Chemistry 2|Physical Chemistry 2|Instrumental Analysis|Polymer Chemistry|Environmental Chemistry|Industrial Chemistry|Chemistry Applications|Chemistry Project'),
'CIV': pairs('مدخل إلى الهندسة المدنية|المساحة 1|الاستاتيكا|المساحة 2|مقاومة المواد|جيولوجيا هندسية|تحليل إنشائي 1|ميكانيكا التربة|تحليل إنشائي 2|هيدروليكا|تصميم الخرسانة|هندسة النقل|تصميم المنشآت المعدنية|هندسة المياه|إدارة التشييد|مشروع مدني', 'Introduction to Civil Engineering|Surveying 1|Statics|Surveying 2|Strength of Materials|Engineering Geology|Structural Analysis 1|Soil Mechanics|Structural Analysis 2|Hydraulics|Concrete Design|Transportation Engineering|Steel Structure Design|Water Engineering|Construction Management|Civil Project'),
'ELE': pairs('مدخل إلى الهندسة الكهربائية|دوائر كهربائية 1|دوائر كهربائية 2|المنطق الرقمي|إلكترونيات 1|إشارات ونظم|إلكترونيات 2|كهرومغناطيسية|نظم التحكم|آلات كهربائية|نظم القدرة|اتصالات تماثلية|إلكترونيات القدرة|اتصالات رقمية|الأنظمة المضمنة|مشروع كهربائي', 'Introduction to Electrical Engineering|Electric Circuits 1|Electric Circuits 2|Digital Logic|Electronics 1|Signals and Systems|Electronics 2|Electromagnetics|Control Systems|Electrical Machines|Power Systems|Analog Communications|Power Electronics|Digital Communications|Embedded Systems|Electrical Project'),
'MEC': pairs('مدخل إلى الهندسة الميكانيكية|استاتيكا هندسية|ديناميكا هندسية|الديناميكا الحرارية 1|ميكانيكا الموائع|الديناميكا الحرارية 2|انتقال الحرارة|عمليات التصنيع|تصميم الآلات 1|اهتزازات ميكانيكية|تصميم الآلات 2|نظم التبريد|محطات القدرة|التحكم الميكانيكي|تصميم ميكانيكي تطبيقي|مشروع ميكانيكي', 'Introduction to Mechanical Engineering|Engineering Statics|Engineering Dynamics|Thermodynamics 1|Fluid Mechanics|Thermodynamics 2|Heat Transfer|Manufacturing Processes|Machine Design 1|Mechanical Vibrations|Machine Design 2|Refrigeration Systems|Power Plants|Mechanical Control|Applied Mechanical Design|Mechanical Project'),
'NUR': pairs('مقدمة في التمريض|المهارات التمريضية 1|أساسيات التمريض|المهارات التمريضية 2|تمريض البالغين 1|التقييم الصحي|تمريض البالغين 2|تمريض الأسرة|تمريض الأمومة|تمريض الأطفال|التمريض النفسي|تمريض المجتمع|التمريض الحرج|إدارة التمريض|الممارسة التمريضية|مشروع تمريض', 'Introduction to Nursing|Nursing Skills 1|Nursing Fundamentals|Nursing Skills 2|Adult Nursing 1|Health Assessment|Adult Nursing 2|Family Nursing|Maternal Nursing|Pediatric Nursing|Mental Health Nursing|Community Nursing|Critical Care Nursing|Nursing Management|Nursing Practice|Nursing Project'),
'MLS': pairs('مقدمة في المختبرات الطبية|تقنيات المختبر|كيمياء سريرية 1|علم الدم 1|كيمياء سريرية 2|علم الدم 2|بكتيريا طبية|طفيليات طبية|علم المناعة السريري|نقل الدم|علم الفيروسات|علم الأنسجة المرضي|التشخيص الجزيئي|إدارة المختبرات|التطبيق المختبري|مشروع مختبرات', 'Introduction to Medical Laboratories|Laboratory Techniques|Clinical Chemistry 1|Hematology 1|Clinical Chemistry 2|Hematology 2|Medical Bacteriology|Medical Parasitology|Clinical Immunology|Blood Transfusion|Virology|Histopathology|Molecular Diagnostics|Laboratory Management|Laboratory Practice|Medical Laboratory Project'),
'PHT': pairs('مقدمة في العلاج الطبيعي|حركة الجسم|العلاج بالتمارين 1|الميكانيكا الحيوية|العلاج بالتمارين 2|الفحص العضلي|العلاج الكهربائي|علاج العظام|التأهيل العصبي|التأهيل القلبي|علاج طبيعي للأطفال|الإصابات الرياضية|التأهيل المجتمعي|العلاج اليدوي|الممارسة التأهيلية|مشروع علاج طبيعي', 'Introduction to Physiotherapy|Kinesiology|Exercise Therapy 1|Biomechanics|Exercise Therapy 2|Muscle Assessment|Electrotherapy|Orthopedic Therapy|Neurological Rehabilitation|Cardiac Rehabilitation|Pediatric Physiotherapy|Sports Injuries|Community Rehabilitation|Manual Therapy|Rehabilitation Practice|Physiotherapy Project'),
'MED': pairs('التفكير الطبي|مصطلحات طبية|الجهاز العضلي الهيكلي|الجهاز القلبي الوعائي|الجهاز التنفسي|الجهاز الهضمي|الجهاز العصبي|الغدد الصماء|آليات المرض|العلاج الدوائي|التشخيص السريري|الفحص السريري|الطب الباطني 1|الجراحة 1|الطب الباطني 2|الجراحة 2|طب الأطفال 1|النساء والولادة 1|طب الأطفال 2|النساء والولادة 2|طب الطوارئ|طب الأسرة|مراجعة سريرية تكاملية|مشروع طب', 'Medical Reasoning|Medical Terminology|Musculoskeletal System|Cardiovascular System|Respiratory System|Digestive System|Nervous System|Endocrine System|Disease Mechanisms|Drug Therapy|Clinical Diagnosis|Clinical Examination|Internal Medicine 1|Surgery 1|Internal Medicine 2|Surgery 2|Pediatrics 1|Obstetrics and Gynecology 1|Pediatrics 2|Obstetrics and Gynecology 2|Emergency Medicine|Family Medicine|Integrated Clinical Review|Medical Project'),
'ARB': pairs('الأدب الجاهلي|مهارات القراءة|الأدب الإسلامي|المعاجم العربية|الأدب الأموي|فقه اللغة|الأدب العباسي|علم الدلالة|الأدب الأندلسي|اللسانيات العامة|الأدب الحديث|علم النص|السرد العربي|النقد الحديث|تحليل النصوص|مشروع لغة عربية', 'Pre Islamic Literature|Reading Skills|Early Islamic Literature|Arabic Lexicography|Umayyad Literature|Arabic Philology|Abbasid Literature|Semantics|Andalusian Literature|General Linguistics|Modern Literature|Text Linguistics|Arabic Narrative|Modern Criticism|Text Analysis|Arabic Project'),
'PHR': pairs('مدخل إلى الصيدلة|مصطلحات صيدلية|صيدلانيات 1|كيمياء دوائية 1|صيدلانيات 2|كيمياء دوائية 2|علم الأدوية 1|عقاقير 1|علم الأدوية 2|عقاقير 2|صيدلة سريرية 1|تكنولوجيا الدواء|صيدلة سريرية 2|تفاعلات دوائية|علاجيات 1|صيدلة المجتمع|علاجيات 2|صيدلة المستشفيات|الرعاية الصيدلية|مشروع صيدلة', 'Introduction to Pharmacy|Pharmacy Terminology|Pharmaceutics 1|Medicinal Chemistry 1|Pharmaceutics 2|Medicinal Chemistry 2|Pharmacology 1|Pharmacognosy 1|Pharmacology 2|Pharmacognosy 2|Clinical Pharmacy 1|Pharmaceutical Technology|Clinical Pharmacy 2|Drug Interactions|Therapeutics 1|Community Pharmacy|Therapeutics 2|Hospital Pharmacy|Pharmaceutical Care|Pharmacy Project'),
}

GENERAL = [
 ('GEN1101','مهارات اللغة الإنجليزية 1','English Language Skills 1',2,1),
 ('GEN1102','مهارات التعلم الجامعي','University Learning Skills',2,1),
 ('GEN1201','مهارات اللغة الإنجليزية 2','English Language Skills 2',2,2),
 ('GEN1202','مهارات الاتصال','Communication Skills',2,2),
 ('QR2103','حفظ القرآن الكريم 2','Quran Memorization 2',2,3),
 ('GEN2302','الثقافة الإسلامية','Islamic Culture',2,3),
 ('GEN2401','الكتابة الأكاديمية','Academic Writing',2,4),
 ('GEN2402','المواطنة والمسؤولية الاجتماعية','Citizenship and Social Responsibility',2,4),
 ('GEN3501','ريادة الأعمال والابتكار','Entrepreneurship and Innovation',2,5),
 ('GEN3502','أخلاقيات المهنة','Professional Ethics',2,5),
 ('GEN3601','مهارات البحث العلمي','Research Skills',2,6),
 ('GEN3602','التنمية المستدامة','Sustainable Development',2,6),
 ('GEN4701','مهارات الاستعداد المهني','Career Preparation',2,7),
 ('GEN4702','التطوع وخدمة المجتمع','Volunteering and Community Service',2,7),
 ('GEN4801','التواصل المهني','Professional Communication',2,8),
 ('GEN4802','إدارة المسار المهني','Career Management',2,8),
]
MAJOR_INFO = {m[0]:(ck, m[1], m[2], m[3]) for ck,(_,_,_,ms) in COLLEGES.items() for m in ms}

# Only ABM name is observed in the source; the other names are explicitly fictional.
CAMPUS = {
 'ABM': 'العابدية - طلاب',
 'MKF': 'مقر مكة الافتراضي - طالبات',
 'JMM': 'مقر الجموم الافتراضي - طلاب',
 'JMF': 'مقر الجموم الافتراضي - طالبات',
 'LTM': 'مقر الليث الافتراضي - طلاب',
 'LTF': 'مقر الليث الافتراضي - طالبات',
}
CAMPUS_CODE = {v:k for k,v in CAMPUS.items()}


def catalog():
    rows, meta, curriculum = [], {}, {}
    def add(code, ar, en, ck, dept, credits, level, lab=False):
        if code in meta:
            return
        college = COLLEGES[ck][0] if ck in COLLEGES else 'متطلبات الجامعة المشتركة'
        rows.append([code,ar,en,college,dept,credits,level])
        meta[code] = dict(ar=ar,en=en,ck=ck,credits=credits,level=level,lab=lab)
    for ck,(college,building,labp,majors) in COLLEGES.items():
        for i,(ar,en) in enumerate(SHARED[ck]):
            level=i//2+1
            code=f'SYN{ck}{level:02d}10{i%2+1}'
            if ck=='CMP' and level==3 and i%2==0:
                code,ar,en='SEC1101','مقدمة في الأمن السيبراني','Introduction to Cybersecurity'
            if ck=='CMP' and level==5:
                code,ar,en = [('AI3301','تحليل وتصوير البيانات','Data Analysis and Visualization'),('HCI2011','تصميم واجهة المستخدم','User Interface Design')][i%2]
            # Only specific applied shared modules carry a separate practical.
            lab = (ck=='CMP' and (code=='HCI2011' or level in (2,4)))
            add(code,ar,en,ck,'المقررات المشتركة بالكلية',3,level,lab)
        for major,mar,_,_ in majors:
            for i,(ar,en) in enumerate(SPECIAL[major]):
                level=i//2+1
                code=f'SYN{major}{level:02d}20{i%2+1}'
                if major=='AI' and level==5:
                    code=['AI3101','AI3201'][i%2]
                # Fixed deterministic lab pattern: reproducible and disclosed.
                lab = labp>0 and ((i*7+len(major))%10 < round(labp*10))
                credits = 4 if ck in ('MED','PHR') and level>=4 and i%2==0 else 3
                add(code,ar,en,ck,mar,credits,level,lab)
            maxlevel=len(SPECIAL[major])//2
            for level in range(1,maxlevel+1):
                shared=[]
                for row in rows:
                    if row[3]==college and row[4]=='المقررات المشتركة بالكلية' and row[6]==level:
                        shared.append(row[0])
                specialized=[r[0] for r in rows if r[4]==mar and r[6]==level]
                assert len(shared)==2 and len(specialized)==2, (major,level,shared,specialized)
                curriculum[(major,level)]=shared+specialized
    for code,ar,en,credits,level in GENERAL:
        add(code,ar,en,'GEN','متطلبات الجامعة',credits,level,False)
    return pd.DataFrame(rows,columns=SCHEMAS['courses']),meta,curriculum


def students(rng):
    rows=[]
    for ck,(_,_,_,majors) in COLLEGES.items():
        for major,name,count,female_share in majors:
            levels=len(SPECIAL[major])//2
            weights=np.linspace(1.15,.85,levels); weights/=weights.sum()
            for _ in range(count):
                gender='Female' if rng.random()<female_share else 'Male'
                level=int(rng.choice(np.arange(1,levels+1),p=weights))
                if ck in ('MED','PHR','AHS'):
                    site='MAIN'
                else:
                    site=str(rng.choice(['MAIN','JUM','LITH'],p=[.84,.10,.06] if ck not in ('CMP','ENG') else [.94,.04,.02]))
                code=('MKF' if gender=='Female' else 'ABM') if site=='MAIN' else (('JMF' if gender=='Female' else 'JMM') if site=='JUM' else ('LTF' if gender=='Female' else 'LTM'))
                rows.append([None,COLLEGES[ck][0],name,name,level,CAMPUS[code],gender])
    assert len(rows)==N_STUDENTS
    order=rng.permutation(N_STUDENTS)
    rows=[rows[int(i)] for i in order]
    for i,row in enumerate(rows,1): row[0]=f'STU{i:06d}'
    return pd.DataFrame(rows,columns=SCHEMAS['students'])


def blockmask(day,start,duration):
    return ((1<<duration)-1) << (day*12+start)


def plan_bundle(rng,codes,meta,days_count,profile):
    """Weighted, nonoverlapping cohort master plan; students later select subsets."""
    bounds={'Morning':(0,6),'Daytime':(2,9),'Evening':(6,12),'Mixed':(0,12)}
    low,high=bounds[profile]
    needs=[]
    for code in codes:
        cr=meta[code]['credits']
        durations=[2,1] if cr==3 else ([2,2] if cr==4 else [cr])
        needs.append((code,'Theory',durations))
        if meta[code]['lab']: needs.append((code,'Practical',[2]))
    total=sum(sum(ds) for _,_,ds in needs)
    if total>(high-low)*days_count-1:
        low,high=0,12
    if total>12*days_count-1: days_count=min(5,days_count+1)
    # Prefer assigning long/double meetings first to reduce fragmentation.
    needs.sort(key=lambda x:(-max(x[2]),-len(x[2]),rng.random()))
    for attempt in range(100):
        ds=sorted(map(int,rng.choice(5,size=days_count,replace=False)))
        mask=0; assigned={}; dayload=Counter()
        for code,activity,durations in needs:
            chosen=[]; used=set()
            for duration in durations:
                candidates=[]; weights=[]
                for day in ds:
                    if day in used: continue
                    for start in range(low,high-duration+1):
                        bm=blockmask(day,start,duration)
                        if mask & bm: continue
                        # morning demand > evening, with lower noon occupancy.
                        hour=8+start
                        hourly=[1.9,1.7,2.0,1.6,.65,1.35,1.20,1.0,.80,.70,.50,.35][start]
                        balance=1/(1+dayload[day]*.75)
                        candidates.append((day,start,duration,bm))
                        weights.append(hourly*balance)
                if not candidates: break
                p=np.asarray(weights); p/=p.sum()
                day,start,duration,bm=candidates[int(rng.choice(len(candidates),p=p))]
                mask|=bm; used.add(day);dayload[day]+=duration
                chosen.append((day,start,duration))
            if len(chosen)!=len(durations): break
            assigned[(code,activity)]=chosen
        else:
            if len(dayload)==days_count:
                return assigned
    # Safe explicit fallback with five unrestricted days, never relax overlaps.
    if profile!='Mixed' or days_count<5:
        return plan_bundle(rng,codes,meta,5,'Mixed')
    raise RuntimeError('Unable to schedule a conflict-free cohort')


class ResourcePool:
    """Reuse rooms and instructors only if all meeting periods are free."""
    def __init__(self):
        self.rooms=defaultdict(list); self.teachers=defaultdict(list)
        self.room_serial=defaultdict(int); self.instructor_serial=0
        self.room_capacity={}; self.room_count=0

    def allocate(self,campus,building,activity,capacity,plan,online,ck):
        mask=0
        for d,s,n in plan: mask |= blockmask(d,s,n)
        code=CAMPUS_CODE[campus]
        if online:
            building='ONLINE'; room=f'SYN-{code}::ONLINE'
        else:
            key=(campus,building,activity,capacity)
            pool=self.rooms[key]
            free=next((r for r in pool if not r[1]&mask),None)
            if free is None:
                self.room_serial[(campus,building)]+=1
                num=self.room_serial[(campus,building)]
                floor=(num-1)//40; number=(num-1)%40+1
                room=f'SYN-{code}::{building}-{floor}.{number:03d}'
                free=[room,0];pool.append(free);self.room_count+=1
                self.room_capacity[room]=capacity
            free[1]|=mask; room=free[0]
        teacherkey=(campus,ck,activity)
        tp=self.teachers[teacherkey]
        load=sum(n for _,_,n in plan)
        teacher=next((t for t in tp if not t[1]&mask and t[2]+load<=18),None)
        if teacher is None:
            self.instructor_serial+=1
            teacher=[f'SYNINS{self.instructor_serial:06d}',0,0];tp.append(teacher)
        teacher[1]|=mask;teacher[2]+=load
        return building,room,teacher[0]


def generate(out: Path,rng,student_df,course_df,meta,curriculum):
    pools=ResourcePool(); section_rows=[]; meeting_rows=[]; enrollment_rows=[]
    course_section_no=Counter(); family_id=0; group_count=0
    lookup={v[1]:k for k,v in MAJOR_INFO.items()}
    grouped=student_df.groupby(['Campus','Major','Academic_Level'],sort=True)
    for (campus,major_name,level),data in grouped:
        major=lookup[major_name];ck=MAJOR_INFO[major][0]
        ids=data.Student_ID.to_numpy().copy();rng.shuffle(ids)
        offset=0
        while offset<len(ids):
            size=int(rng.choice([30,36,42,48],p=[.14,.23,.36,.27]))
            if 0<len(ids)-offset-size<12: size=(len(ids)-offset+1)//2
            members=ids[offset:offset+size];offset+=len(members);n=len(members)
            group_count+=1
            specialized=curriculum[(major,int(level))]
            gens=[r[0] for r in GENERAL if r[4]==min(int(level),8)]
            codes=specialized+gens
            daycount=int(rng.choice([2,3,4,5],p=[.06,.22,.45,.27]))
            profile=str(rng.choice(['Morning','Daytime','Evening','Mixed'],p=[.26,.35,.10,.29]))
            if daycount==2: profile='Mixed'
            plan=plan_bundle(rng,codes,meta,daycount,profile)
            # Student-level choices are made AFTER offerings and their times exist.
            choices=[]
            for i,sid in enumerate(members):
                selected={specialized[0],specialized[2],gens[0]}
                if rng.random()<.82: selected.add(specialized[1])
                if rng.random()<.70: selected.add(specialized[3])
                if rng.random()<.48: selected.add(gens[1])
                choices.append(selected)
            # Keep every planned offering meaningful, including very small cohorts.
            for j,code in enumerate(codes):
                if not any(code in c for c in choices): choices[j%n].add(code)

            for code in codes:
                family_id+=1
                family=f'SYNSEC{family_id:06d}'
                capacity=next(c for c in (30,40,50,60) if c>=n)
                online= code=='QR2103' or (meta[code]['ck']=='GEN' and rng.random()<.24) or (not meta[code]['lab'] and rng.random()<.035)
                mode='Online' if online else 'In-person'
                building=str(COLLEGES[ck][1])
                if code=='HCI2011': building='23'
                b,room,inst=pools.allocate(campus,building,'Theory',capacity,plan[(code,'Theory')],online,ck)
                sec=family+'-T'
                course_section_no[(code,'Theory')]+=1
                number=course_section_no[(code,'Theory')]
                section_rows.append([sec,code,number,'Theory',inst,capacity,campus,b,room,mode])
                for day,start,dur in plan[(code,'Theory')]:
                    meeting_rows.append([f'SYNMTG{len(meeting_rows)+1:07d}',sec,DAYS[day],f'{8+start:02d}:00',f'{8+start+dur-1:02d}:50',campus,b,room,mode])
                enrolled=[sid for sid,c in zip(members,choices) if code in c]
                enrollment_rows.extend([sid,sec,code] for sid in enrolled)
                if meta[code]['lab']:
                    # Practical groups cap at 24; each student gets exactly one linked practical.
                    for gi,starti in enumerate(range(0,len(enrolled),24),1):
                        lab_members=enrolled[starti:starti+24]
                        labsec=f'{family}-P{gi:02d}'
                        b,room,inst=pools.allocate(campus,'25' if code=='HCI2011' else building,'Practical',24,plan[(code,'Practical')],False,ck)
                        course_section_no[(code,'Practical')]+=1
                        section_rows.append([labsec,code,course_section_no[(code,'Practical')],'Practical',inst,24,campus,b,room,'In-person'])
                        for day,start,dur in plan[(code,'Practical')]:
                            meeting_rows.append([f'SYNMTG{len(meeting_rows)+1:07d}',labsec,DAYS[day],f'{8+start:02d}:00',f'{8+start+dur-1:02d}:50',campus,b,room,'In-person'])
                        enrollment_rows.extend([sid,labsec,code] for sid in lab_members)
            if group_count%250==0:
                print(f'  Scheduled {group_count:,} cohorts, {len(section_rows):,} sections',flush=True)
    sections=pd.DataFrame(section_rows,columns=SCHEMAS['sections'])
    meetings=pd.DataFrame(meeting_rows,columns=SCHEMAS['section_meetings'])
    enroll=pd.DataFrame(enrollment_rows,columns=SCHEMAS['student_enrollments']).sort_values(['Student_ID','Section_ID'],kind='stable').reset_index(drop=True)
    # Persist normalized tables first, then derive individual timetables by joins.
    for name,df in [('students',student_df),('courses',course_df),('sections',sections),('section_meetings',meetings),('student_enrollments',enroll)]:
        df.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    section_times=sections.merge(meetings.drop(columns=['Campus','Building','Room','Delivery_Mode']),on='Section_ID',validate='one_to_many')
    section_times=section_times.merge(course_df[['Course_Code','Course_Name_AR']],on='Course_Code',validate='many_to_one').rename(columns={'Course_Name_AR':'Course_Name'})
    count=0
    with (out/'student_timetables.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        for start in range(0,N_STUDENTS,2000):
            batch=student_df.iloc[start:start+2000]
            lo,hi=batch.Student_ID.iloc[0],batch.Student_ID.iloc[-1]
            e=enroll.loc[enroll.Student_ID.between(lo,hi)]
            data=e[['Student_ID','Section_ID']].merge(section_times,on='Section_ID',validate='many_to_many')
            data=data.merge(batch[['Student_ID','College','Major','Academic_Level']],on='Student_ID',validate='many_to_one')
            data['_day']=data.Day.map(DAY_ORDER)
            data=data.sort_values(['Student_ID','_day','Start_Time','Section_ID'],kind='stable')
            data[SCHEMAS['student_timetables']].to_csv(handle,index=False,header=start==0)
            count+=len(data)
    print(f'Generated {count:,} individual weekly meeting rows',flush=True)
    return dict(cohorts=group_count,rooms=pools.room_count,instructors=pools.instructor_serial,rows=count)


def minutes(s):
    return s.str[:2].astype('int32')*60+s.str[3:5].astype('int32')


def overlaps(df,keys):
    d=df.copy();d['_start']=minutes(d.Start_Time);d['_end']=minutes(d.End_Time)
    d=d.sort_values(keys+['_start','_end'])
    prior=d.groupby(keys,sort=False)._end.cummax().groupby([d[k] for k in keys],sort=False).shift()
    return int((d._start<prior).sum())


def validate(out: Path,meta,curriculum):
    """Read all CSVs back from disk. Stream the large denormalized table."""
    checks=[]
    def check(name,errors,detail):
        checks.append([name,'PASS' if int(errors)==0 else 'FAIL',int(errors),detail])
    tables={name:pd.read_csv(out/f'{name}.csv',encoding='utf-8-sig',dtype=str,keep_default_na=False) for name in ['students','courses','sections','section_meetings','student_enrollments']}
    s,c,sc,m,e=(tables[n] for n in ['students','courses','sections','section_meetings','student_enrollments'])
    for name,df in tables.items():
        check(f'{name} schema',0 if list(df.columns)==SCHEMAS[name] else 1,'Exact requested column names and order')
        check(f'{name} mandatory values',int((df=='').sum().sum()),'No empty fields; Online rooms use explicit virtual-room identifiers')
    check('Exactly 59000 students',abs(len(s)-N_STUDENTS),f'Rows={len(s):,}')
    check('Unique student IDs',s.Student_ID.duplicated().sum(),'No Student_ID assigned twice')
    check('Synthetic student identifier range',int((~s.Student_ID.str.fullmatch(r'STU\d{6}')).sum())+len(set(s.Student_ID)^set(f'STU{i:06d}' for i in range(1,N_STUDENTS+1))),'Expected STU000001 through STU059000')
    for df,col in [(c,'Course_Code'),(sc,'Section_ID'),(m,'Meeting_ID')]:
        check(f'Unique {col}',df[col].duplicated().sum(),'Primary key uniqueness')
    check('Unique student-section registrations',e.duplicated(['Student_ID','Section_ID']).sum(),'No duplicate enrollment')
    check('Section course foreign keys',(~sc.Course_Code.isin(c.Course_Code)).sum(),'Every offering references an existing course')
    check('Meeting section foreign keys',(~m.Section_ID.isin(sc.Section_ID)).sum(),'Every meeting references an existing offering')
    check('Enrollment foreign keys',int((~e.Student_ID.isin(s.Student_ID)).sum()+(~e.Section_ID.isin(sc.Section_ID)).sum()+(~e.Course_Code.isin(c.Course_Code)).sum()),'Student, section and course all exist')
    ec=e.merge(sc[['Section_ID','Course_Code','Capacity','Campus','Activity_Type']],on='Section_ID',suffixes=('_reg','_sec'),validate='many_to_one')
    check('Enrollment course equals section course',(ec.Course_Code_reg!=ec.Course_Code_sec).sum(),'No mismatched course on registration')
    counts=e.groupby('Section_ID').Student_ID.nunique()
    cap=sc.set_index('Section_ID').Capacity.astype(int)
    check('Section capacities',int((counts>cap.reindex(counts.index)).sum()),f'All {len(sc):,} sections within capacity')
    check('All sections used',int((~sc.Section_ID.isin(e.Section_ID)).sum()),'No empty generated offerings')
    check('All sections have meetings',int((~sc.Section_ID.isin(m.Section_ID)).sum()),'Every section has at least one weekly meeting')
    check('All catalogue courses offered',int((~c.Course_Code.isin(sc.Course_Code)).sum()),'Every catalogue item has an offering in the dataset')
    check('All students enrolled',len(set(s.Student_ID)-set(e.Student_ID)),'Every student registered in at least one course')
    mm=m.merge(sc,on='Section_ID',suffixes=('_meeting','_section'),validate='many_to_one')
    mismatch=0
    for field in ['Campus','Building','Room','Delivery_Mode']:
        mismatch+=int((mm[field+'_meeting']!=mm[field+'_section']).sum())
    check('Meeting and section location consistency',mismatch,'Campus, building, global room ID and delivery mode match')
    check('Permitted days',(~m.Day.isin(DAYS)).sum(),'Sunday through Thursday only')
    time_pattern=r'(?:0[8-9]|1[0-9]):(?:00|50)'
    badtime=(~m.Start_Time.str.fullmatch(time_pattern))|(~m.End_Time.str.fullmatch(time_pattern))
    start,end=minutes(m.Start_Time),minutes(m.End_Time)
    badtime|=(start<480)|(end>1200)|(start>=end)|(~m.Start_Time.str.endswith(':00'))|(~m.End_Time.str.endswith(':50'))
    check('Time boundaries and period grid',badtime.sum(),'08:00 through 19:50; start :00 and end :50; positive duration')
    physical=mm[mm.Delivery_Mode_section=='In-person'].copy()
    physical['Room']=physical.Room_section
    check('Room booking conflicts',overlaps(physical,['Room','Day']),'All physical-room intervals checked; online virtual rooms are not physical space')
    check('Instructor timetable conflicts',overlaps(mm,['Instructor_ID','Day']),'No lecturer teaches simultaneous meetings')
    check('Instructor campus consistency',(sc.groupby('Instructor_ID').Campus.nunique()>1).sum(),'Each generated instructor belongs to one campus')
    lengths=((minutes(mm.End_Time)-minutes(mm.Start_Time)+10)//60).astype(int)
    load=lengths.groupby(mm.Instructor_ID).sum()
    check('Instructor weekly load limit',int((load>18).sum()),f'Maximum={load.max()} period slots; assumed upper bound=18')
    per_section_periods=lengths.groupby(mm.Section_ID).sum()
    expected_periods=sc.set_index('Section_ID').apply(lambda r:2 if r.Activity_Type=='Practical' else meta[r.Course_Code]['credits'],axis=1)
    check('Course contact period consistency',int((per_section_periods!=expected_periods.reindex(per_section_periods.index)).sum()),'Theory period totals equal catalogue credits; practical sections have two periods')
    check('Permitted activity types',int((~sc.Activity_Type.isin(['Theory','Practical'])).sum()),'Explicit separate theory and practical offerings')
    check('Permitted delivery modes',int((~sc.Delivery_Mode.isin(['In-person','Online'])).sum()),'Only In-person and Online')
    online=sc.Delivery_Mode=='Online'
    check('Online location semantics',int(((sc.Building=='ONLINE')!=online).sum()+((sc.Room.str.endswith('::ONLINE'))!=online).sum()),'Online locations are explicit virtual identifiers, never physical room reservations')
    check('Physical room seat consistency',int((sc[~online].groupby('Room').Capacity.nunique()>1).sum()),'Every synthetic physical room retains one modelled seating capacity across all bookings')
    check('Room IDs globally unique by campus',int((sc.groupby('Room').Campus.nunique()>1).sum()),'Room keys contain SYN campus prefix plus PDF-like room code')
    check('Physical room code pattern',int((~physical.Room.str.fullmatch(r'SYN-[A-Z]{3}::\d+-\d+\.\d{3}')).sum()),'Example: SYN-ABM::12-0.018; all generated rooms are fictional')
    stu_campus=s.set_index('Student_ID').Campus
    check('Student-section campus consistency',int((ec.Campus.to_numpy()!=ec.Student_ID.map(stu_campus).to_numpy()).sum()),'No cross-campus student travel in this scenario')
    check('Campus gender consistency',int(((s.Gender=='Male')&s.Campus.str.endswith('طالبات')).sum()+((s.Gender=='Female')&~s.Campus.str.endswith('طالبات')).sum()),'Male and female campus labels match each synthetic student')
    theor=ec[ec.Activity_Type=='Theory']
    check('One theory section per student-course',theor.duplicated(['Student_ID','Course_Code_reg']).sum(),'No multiple theory enrollments for the same course')
    pr=ec[ec.Activity_Type=='Practical'].copy()
    pr['Parent']=pr.Section_ID.str.replace(r'-P\d+$','-T',regex=True)
    expected_theory=pd.MultiIndex.from_frame(e[['Student_ID','Section_ID']])
    check('Practical linked to enrolled theory',int((~pd.MultiIndex.from_frame(pr[['Student_ID','Parent']]).isin(expected_theory)).sum()),'SYNSECxxxxxx-Pnn links to SYNSECxxxxxx-T')
    check('No duplicate practical per student-course',pr.duplicated(['Student_ID','Course_Code_reg']).sum(),'Exactly one practical group when required')
    lab_theory=theor[theor.Course_Code_reg.map(lambda x:meta[x]['lab'])]
    a=set(zip(lab_theory.Student_ID,lab_theory.Course_Code_reg)); b=set(zip(pr.Student_ID,pr.Course_Code_reg))
    check('Required practical coverage',len(a^b),'Students in a lab-bearing course receive exactly one linked practical')
    practical_parent=sc[sc.Activity_Type=='Practical'].copy()
    practical_parent['Parent']=practical_parent.Section_ID.str.replace(r'-P\d+$','-T',regex=True)
    parent_map=sc.set_index('Section_ID')
    link_errors=int((practical_parent.Course_Code!=practical_parent.Parent.map(parent_map.Course_Code)).sum()+(practical_parent.Campus!=practical_parent.Parent.map(parent_map.Campus)).sum())
    check('Practical parent metadata',link_errors,'Practical and parent theory have the same academic course and campus')
    info=s.set_index('Student_ID')[['Major','Academic_Level']].to_dict('index')
    reverse={v[1]:k for k,v in MAJOR_INFO.items()}
    allowed={(v[1],level):set(curriculum[(major,level)]+[g[0] for g in GENERAL if g[4]==min(level,8)]) for major,v in MAJOR_INFO.items() for level in range(1,len(SPECIAL[major])//2+1)}
    invalid=0
    for row in e.itertuples(index=False):
        st=info[row.Student_ID]
        if row.Course_Code not in allowed[(st['Major'],int(st['Academic_Level']))]: invalid+=1
    check('Major and level course compatibility',invalid,'Only same-level specialty/college courses plus the documented general requirements')
    expected_rows=int(e.Section_ID.map(m.groupby('Section_ID').size()).sum())
    canon=sc.merge(m[['Section_ID','Day','Start_Time','End_Time']],on='Section_ID',validate='one_to_many').merge(c[['Course_Code','Course_Name_AR']],on='Course_Code',validate='many_to_one').rename(columns={'Course_Name_AR':'Course_Name'})
    canon=canon[['Section_ID','Day','Start_Time','End_Time','Course_Code','Section_Number','Activity_Type','Campus','Building','Room','Delivery_Mode','Course_Name']]
    enrollment_keys=pd.MultiIndex.from_frame(e[['Student_ID','Section_ID']])
    student_data=s.set_index('Student_ID')
    errors=Counter();actual_rows=0;seen_students=set();enrollment_meetings=Counter()
    rowhashes=set(); daysets=defaultdict(set); spans={}; course_sets=defaultdict(set); meeting_counts=Counter(); student_bounds={}
    for chunknum,ch in enumerate(pd.read_csv(out/'student_timetables.csv',encoding='utf-8-sig',dtype=str,keep_default_na=False,chunksize=75000)):
        errors['schema']+=list(ch.columns)!=SCHEMAS['student_timetables']
        errors['missing']+=int((ch=='').sum().sum())
        actual_rows+=len(ch); seen_students.update(ch.Student_ID)
        idx=pd.MultiIndex.from_frame(ch[['Student_ID','Section_ID']])
        errors['unregistered']+=int((~idx.isin(enrollment_keys)).sum())
        joined=ch.merge(canon,on=['Section_ID','Day','Start_Time','End_Time'],how='left',suffixes=('_out','_canonical'),validate='many_to_one',indicator=True)
        errors['unknown_meetings']+=int((joined._merge!='both').sum())
        for field in ['Course_Code','Section_Number','Activity_Type','Campus','Building','Room','Delivery_Mode','Course_Name']:
            errors['canonical_mismatch']+=int((joined[field+'_out']!=joined[field+'_canonical']).sum())
        for field in ['College','Major','Academic_Level']:
            errors['student_mismatch']+=int((ch[field]!=ch.Student_ID.map(student_data[field])).sum())
        hashes=pd.util.hash_pandas_object(ch[['Student_ID','Section_ID','Day','Start_Time','End_Time']],index=False).to_numpy()
        for h in hashes:
            value=int(h)
            if value in rowhashes: errors['duplicates']+=1
            rowhashes.add(value)
        for row in ch[['Student_ID','Section_ID','Day','Start_Time','End_Time','Course_Code']].itertuples(index=False,name=None):
            sid,sec,day,st,en,co=row
            t0=int(st[:2])*60+int(st[3:]);t1=int(en[:2])*60+int(en[3:])
            key=(sid,day)
            # Master output is ordered by student/day/start; compare across chunks too.
            if key in spans and t0<spans[key]: errors['student_overlap']+=1
            spans[key]=max(t1,spans.get(key,0))
            bounds=student_bounds.get(sid,(1440,0))
            student_bounds[sid]=(min(t0,bounds[0]),max(t1,bounds[1]))
            daysets[sid].add(day);course_sets[sid].add(co);meeting_counts[sid]+=1
            enrollment_meetings[(sid,sec)]+=1
        if chunknum%4==0: print(f'  Validated {actual_rows:,} timetable rows',flush=True)
    check('Timetable exact schema',errors['schema'],'Exact user-requested columns, including one row per student meeting')
    check('Timetable mandatory values',errors['missing'],'No missing essential timetable values')
    check('Timetable registrations valid',errors['unregistered'],'All timetable rows belong to a registered student-section pair')
    check('Timetable meetings valid',errors['unknown_meetings'],'Every individual meeting exists in the canonical section timetable')
    check('Identical meeting metadata for enrolled students',errors['canonical_mismatch'],'Every published time, course, activity, room and mode matches section master')
    check('Timetable student metadata',errors['student_mismatch'],'College, major and level match students.csv')
    check('No duplicated timetable meetings',errors['duplicates'],'Duplicate row keys checked across all chunks')
    check('No overlapping student meetings',errors['student_overlap'],'All 59,000 timetables checked, including chunk boundaries')
    check('Every student has a timetable',len(set(s.Student_ID)-seen_students),'No student omitted from the denormalized output')
    check('Expected denormalized row count',abs(expected_rows-actual_rows),f'Expected={expected_rows:,}; actual={actual_rows:,}')
    per_section=m.groupby('Section_ID').size().to_dict()
    badmeetingcount=sum(enrollment_meetings[(sid,sec)]!=per_section[sec] for sid,sec in zip(e.Student_ID,e.Section_ID))
    check('Complete weekly meeting expansion',badmeetingcount,'Every enrollment expands to ALL meetings of that section exactly once')
    expected_anchor={'AI3101','AI3201','AI3301','HCI2011','SEC1101','QR2103'}
    check('PDF reference course coverage',len(expected_anchor-set(e.Course_Code)),'All six reference course codes are present and enrolled')
    distribution=dict(sorted(Counter(map(len,daysets.values())).items()))
    check('Two to five attendance days',sum(len(v)<2 or len(v)>5 for v in daysets.values()),json.dumps(distribution))
    check('All attendance-day counts represented',len({2,3,4,5}-set(distribution)),'Students with two, three, four and five attendance days exist')
    academic_profiles=Counter('Morning only' if end<=840 else ('Evening only' if start>=840 else 'Daytime / mixed') for start,end in student_bounds.values())
    check('Morning evening mixed timetable variety',len({'Morning only','Evening only','Daytime / mixed'}-set(academic_profiles)),'Morning-only, evening-only and daytime/mixed schedules exist')
    first_class_hours=Counter(start//60 for start,_ in student_bounds.values())
    check('Different weekly first class times',len({8,10,12,14}-set(first_class_hours)),'Weekly earliest academic meetings include 08:00, 10:00, 12:00 and 14:00; these are NOT predicted campus arrival times')
    bom_errors=0
    for name in tables.keys()|{'student_timetables'}:
        with (out/f'{name}.csv').open('rb') as fh: bom_errors+=fh.read(3)!=b'\xef\xbb\xbf'
    check('CSV UTF8 BOM',bom_errors,'UTF-8-SIG for every generated data CSV')
    report=pd.DataFrame(checks,columns=SCHEMAS['validation_report'])
    report.to_csv(out/'validation_report.csv',index=False,encoding='utf-8-sig')
    summary={
        'rows':{**{k:len(v) for k,v in tables.items()},'student_timetables':actual_rows,'validation_report':len(report)},
        'checks':len(report),'failures':int((report.Result=='FAIL').sum()),
        'error_count':int(report.Error_Count.sum()),
        'colleges':s.College.nunique(),'majors':s.Major.nunique(),'campuses':s.Campus.nunique(),
        'gender':s.Gender.value_counts().to_dict(),
        'college_counts':s.College.value_counts().to_dict(),
        'campus_counts':s.Campus.value_counts().to_dict(),
        'days':distribution,
        'academic_time_profiles':dict(academic_profiles),
        'course_count_range':[min(map(len,course_sets.values())),max(map(len,course_sets.values()))],
        'meetings_range':[min(meeting_counts.values()),max(meeting_counts.values())],
        'online_meetings':int((m.Delivery_Mode=='Online').sum()),
        'physical_rooms':physical.Room.nunique(),
        'instructors':sc.Instructor_ID.nunique(),
    }
    if summary['failures']:
        print(report.loc[report.Result=='FAIL'].to_string(index=False),flush=True)
        raise RuntimeError(f"Validation failed: {summary['failures']} checks")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def readme(out: Path,summary):
    counts='\n'.join(f'| `{k}.csv` | {v:,} |' for k,v in summary['rows'].items())
    college_counts='\n'.join(f'| {k} | {v:,} |' for k,v in summary['college_counts'].items())
    campus_counts='\n'.join(f'| {k} | {v:,} |' for k,v in summary['campus_counts'].items())
    day_counts='\n'.join(f'| {k} | {v:,} |' for k,v in summary['days'].items())
    columns='\n\n'.join(f"### {name}.csv\n\n`{', '.join(cols)}`" for name,cols in SCHEMAS.items())
    content=f'''# UQU 59000 Synthetic Student Timetables

**SYNTHETIC DATA - NOT OFFICIAL UMM AL-QURA UNIVERSITY RECORDS.**

هذه قاعدة بيانات أكاديمية صناعية بالكامل، وليست جداول رسمية أو سجلات أصلية من الجامعة. جميع معرفات الطلاب والمدرسين والشعب والقاعات مولدة. لا توجد أسماء طلاب أو أرقام جامعية حقيقية. لا تتضمن الحزمة بيانات مرور أو مواقف أو توقعات وصول ومغادرة.

## Source PDF analysis

The user-supplied academic-portal timetable, first term 1448 AH, has two pages. All schedule content appears on page 1; page 2 contains only pagination. Its weekly grid runs Sunday to Thursday, with hourly start slots from 08:00 to 19:00 and end labels at :50. A two-period class therefore uses 08:00-09:50 rather than 08:00-10:00. Cards contain course code, Arabic course name, Theory/Practical activity, section number, campus, and room. Recurring course meetings can be on different days. The sample explicitly shows an online Quran meeting and the campus label العابدية - طلاب.

Observed anchors used in the generated catalogue:

| Code | Arabic name observed in PDF | Illustrative English translation |
|---|---|---|
| AI3101 | مبادىء وتقنيات الذكاء الاصطناعي | Principles and Techniques of AI |
| AI3201 | تعلم الآلة | Machine Learning |
| AI3301 | تحليل وتصوير البيانات | Data Analysis and Visualization |
| HCI2011 | تصميم واجهة المستخدم | User Interface Design |
| SEC1101 | مقدمة في الأمن السيبراني | Introduction to Cybersecurity |
| QR2103 | حفظ القرآن الكريم 2 | Quran Memorization 2 |

Observed room-format examples include 12-0.018, 12-0.003, 12-0.021, 23-0.005, 25-0.028, and 12-1.017. These inspire the FORMAT, not a verified room inventory. The generator does not copy an individual's exact weekly schedule. No PDF or personal source file is redistributed in this ZIP. English translations, credits, course levels, all additional course titles and codes, and all generated assignments are modelling assumptions. A course code's numeric digits do not define its generated academic level: use Course_Level.

## Exact output row counts

Counts below exclude the header row.

| File | Data rows |
|---|---:|
{counts}

`README.md` and `generate_dataset.py` are also included. All seven CSVs are UTF-8-SIG with one initial BOM, English column names and HH:MM clock values. Comma is the delimiter. Rows in student_timetables.csv are ordered by Student_ID, weekday order, start time and section.

## Relational model and column guide

students.Student_ID is the student key. courses.Course_Code is the course key. sections.Section_ID identifies ONE ACTIVITY offering. section_meetings is one-to-many from sections. student_enrollments is the many-to-many link between students and offerings. student_timetables expands every enrollment to every weekly meeting, not one row per student or course.

{columns}

### Key meanings and units

- Student_ID: exact synthetic range STU000001-STU059000; never a real university identifier.
- College, Department, Major: Arabic scenario labels, inspired by academic fields; no claim that this is UQU's verified organization chart. Department equals the major's teaching unit in students.csv. Shared courses use a college-common or university-common department.
- Academic_Level and Course_Level: modelled semester/stage index. Most majors use 1-8, Pharmacy 1-10, Medicine 1-12. This is not a verified UQU study plan and does not reconstruct prerequisite histories.
- Credit_Hours: theoretical credit assumption. Three-credit theory courses meet 2+1 periods; four-credit courses meet 2+2; two-credit general courses meet one two-period block. Each period is 50 contact minutes with 10-minute hourly breaks. A separate practical, when assigned, adds one two-period session without adding a second copy of the course credits.
- Section_Number: sequential numeric offering number within Course_Code and Activity_Type across all campuses. Section_ID is the authoritative join key.
- Activity_Type: Theory or Practical. A course may have a theory offering only or both theory and practical offerings.
- Section_ID linkage: `SYNSEC000001-T` is a theory section; `SYNSEC000001-P01`, `-P02`, etc. are its linked practical groups. Each registered student in a practical-bearing course attends the parent theory and exactly ONE practical group. Do not double-count those as two academic courses.
- Instructor_ID: synthetic instructor identifier. Teachers are assigned to one campus with no overlapping meetings and at most 18 scheduled periods a week in this scenario.
- Capacity: student seats in the activity offering. Theory capacities are 30/40/50/60; practical capacity is 24. Room allocation uses the same or larger capacity, and every section has at least one enrollee.
- Day: Sunday, Monday, Tuesday, Wednesday or Thursday only.
- Start_Time and End_Time: local weekly wall-clock times, HH:MM, no actual calendar dates. No class before 08:00 or after 19:50. The gap between periods is included in a multi-period meeting's elapsed interval.
- Campus: home campus label retained for physical and online sections. Every student belongs to one campus for the whole week.
- Building: synthetic building number; ONLINE for virtual delivery.
- Room: globally unique key, e.g. `SYN-ABM::12-0.018`. The suffix follows the PDF style. Prefix distinguishes equally numbered rooms at different campuses. No generated room is claimed as a real physical room, even if its suffix matches the PDF. Online uses `SYN-ABM::ONLINE` (or another campus code) and is excluded from PHYSICAL room-conflict tests.
- Delivery_Mode: In-person or Online. All meetings of a section use the declared mode and location. Online lectures still count toward student and instructor time-conflict checks.
- Course_Name in student_timetables: Arabic catalogue title copied from courses.Course_Name_AR.
- Result and Error_Count: PASS/FAIL and number of offending records or relations for the named validation test; see Details for exact coverage. PASS does not prove external realism.

## Generation assumptions and algorithm

Seed: **{SEED}** using NumPy default_rng. The same seed and same Python/NumPy/Pandas versions reproduce table values. ZIP metadata timestamps are not part of reproducibility.

1. Construct a reusable course catalogue and level-specific curricula first: two shared college modules, two specialty modules, and two eligible general requirements per major/level. General requirements are shared across colleges. Six codes/names are anchored to the PDF. The rest are synthetic. Each major receives only its eligible curriculum and general requirements, not arbitrary cross-major subjects.
2. Generate exactly 59,000 student rows using fixed unequal major counts and random but seeded levels, gender and campus choices. Student IDs are shuffled across academic groups, not encoded with personal attributes.
3. Form demand-aware cohorts within the SAME campus, major and level, with nominal 30/36/42/48 seats. Small remainders may create smaller cohorts. This is a constructive block-registration model, not a solved recreation of the university's real registration system. Students naturally share sections, and some timetables may be identical.
4. Create the section timetable before student enrollment: choose 2-5 weekly days and a Morning/Daytime/Evening/Mixed time preference. Weighted starts favour the morning while retaining afternoon/evening classes and gaps. Schedule every planned meeting with disjoint period bitmasks. A meeting cannot straddle the end of the daily grid. If a narrow profile is infeasible, widen it; never relax no-overlap constraints.
5. Reserve a room and instructor only if ALL section meeting periods are free. Reuse resources when possible. Add an explicitly fictional room/instructor when existing resources are insufficient rather than claiming a fixed official inventory. This has no institutional capacity-calibration meaning. Teachers are not verified UQU staff.
6. Register students into a compatible subset of the offerings: two selected core modules and one general requirement are mandatory in this model; the remaining college/major modules are selected with probabilities 0.82/0.70 and a second general requirement with probability 0.48. Very small groups are adjusted to keep all created offerings used. This creates variable 3-6 course loads, different gaps and weekly meetings while preserving coherent shared sections. It is a realistic-structure synthetic scenario, not a claim about official degree requirements or real enrollment behaviour.
7. For each practical-bearing course, split enrolled students into groups of at most 24. All linked practical groups have the same planned time but different rooms and instructors. A student attends only one such practical.
8. Write normalized CSVs, then derive the large individual timetable by Pandas joins in batches of 2,000 students. No timetable row is invented independently of an actual section meeting and registration.
9. Read CSVs back using Pandas, validate the normalized relationships and the entire individual timetable in chunks of 75,000 rows, and stop without producing a final ZIP if a test fails.

Online assumptions: QR2103 is online as inspired by the example; other general offerings have 24% online probability and other non-lab theory offerings have 3.5% online probability. This is generated scenario design, not UQU delivery-mode statistics.

### Assumed college population allocation

| College label | Synthetic students |
|---|---:|
{college_counts}

Major allocations are in COLLEGES in the source code. They sum exactly to 59,000. The generated scenario has {summary['colleges']} college labels and {summary['majors']} majors.

### Campuses and gender

Only the name العابدية - طلاب is observed in the supplied PDF. All other campus names deliberately include الافتراضي (fictional). They are geographic scenario labels, not verified campus locations or verified program offerings. Technical fields favour the main modelling site; Medicine, Pharmacy and Applied Health stay at the main modelling sites. For other fields, the assumed mix is 84% main, 10% Jumum-labelled and 6% Lith-labelled sites; Computing/Engineering use 94%/4%/2%. Gender shares differ by major and are arbitrary scenario assumptions printed in COLLEGES. No demographic inference is made about real UQU students.

| Campus scenario label | Synthetic students |
|---|---:|
{campus_counts}

### Generated attendance distribution

| Weekly days with classes | Students |
|---|---:|
{day_counts}

Students have {summary['course_count_range'][0]}-{summary['course_count_range'][1]} distinct academic courses and {summary['meetings_range'][0]}-{summary['meetings_range'][1]} weekly meeting rows in this run. Count unique Course_Code per student to count courses, because theory and practical are linked activity registrations for one course. Shared enrollment patterns are normal and do not mean Student_IDs are duplicated. The constructed dataset uses {summary['physical_rooms']:,} physical virtual-inventory room keys and {summary['instructors']:,} synthetic instructors.

## Validation coverage and limitations

**{summary['checks']} programmed tests passed, {summary['failures']} failed checks, {summary['error_count']} reported errors.** The included validation_report.csv is the per-test evidence. Checks cover the exact population, keys, foreign keys, schemas, completeness, section capacities, physical room conflicts, instructor conflicts, student overlaps, matching section schedules, complete weekly expansion, eligible curricula, practical linkage, day/time rules, and UTF-8 BOMs.

No real student records were used. No student name, national ID, phone number or email is included. Every CSV inherits this synthetic status from its identifiers and this README. Official degree-plan prerequisites, staff contracts, special-room equipment, holidays, gender policy, room inventories and geographic walking times are not calibrated. All physical classes for each student remain on one campus, but building-to-building walking distances are not modelled. A 10-minute period break is the only assumed adjacent-class transfer time. The source is a SINGLE timetable and cannot establish university-wide timetable frequencies. Population size 59,000 is a user-requested scenario size, not an independently verified institutional statistic. No downstream research result should represent these data as observed or official UQU records.

## Run and validate

Install dependencies if needed:

```bash
python -m pip install pandas numpy
python generate_dataset.py --output-dir UQU_59000_Student_Timetables --seed {SEED}
python generate_dataset.py --output-dir UQU_59000_Student_Timetables --validate-only
```

The script writes the seven CSVs, this README, and a copy of itself into the requested output folder. It then creates UQU_59000_Student_Timetables.zip in that folder's parent. Do not use a folder containing unrelated files as the output. The ZIP deliberately contains exactly the nine requested deliverables.

Example Pandas access:

```python
import pandas as pd
students = pd.read_csv('students.csv', encoding='utf-8-sig')
for chunk in pd.read_csv('student_timetables.csv', encoding='utf-8-sig', chunksize=100_000):
    sample = chunk[chunk['Student_ID'] == 'STU000001']
    if not sample.empty:
        print(sample)
```

The large CSV should be processed in chunks rather than opened as a styled workbook. There are no formulas hidden in the CSVs and no executable content other than the requested reproducible Python source.
'''
    (out/'README.md').write_text(content,encoding='utf-8')


def package_dataset(out: Path):
    """Close and CRC-check a temporary ZIP before atomically publishing it."""
    files=[f'{name}.csv' for name in SCHEMAS]+['README.md','generate_dataset.py']
    assert set(files)=={x.name for x in out.iterdir()},'Unexpected deliverables in output folder'
    archive=out.parent/'UQU_59000_Student_Timetables.zip'
    temporary=out.parent/'UQU_59000_Student_Timetables.zip.partial'
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for filename in files: z.write(out/filename,arcname=filename)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert z.namelist()==files
    temporary.replace(archive)
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='UQU_59000_Student_Timetables')
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();out=Path(args.output_dir).resolve()
    if args.seed!=SEED:
        raise SystemExit(f'This audited release uses seed {SEED}; edit the source constant explicitly to make another release.')
    course_df,meta,curriculum=catalog()
    if args.validate_only:
        validate(out,meta,curriculum);return
    out.mkdir(parents=True,exist_ok=True)
    expected={f'{k}.csv' for k in SCHEMAS}|{'README.md','generate_dataset.py'}
    unrelated=[x.name for x in out.iterdir() if x.name not in expected]
    if unrelated: raise SystemExit(f'Output folder contains unrelated entries; choose a clean folder: {unrelated}')
    t=time.time();rng=np.random.default_rng(SEED)
    print(f'Catalogue: {len(course_df)} synthetic/reference-inspired courses',flush=True)
    student_df=students(rng)
    stats=generate(out,rng,student_df,course_df,meta,curriculum)
    print('Reading all exported data back and validating...',flush=True)
    summary=validate(out,meta,curriculum)
    readme(out,summary)
    source=Path(__file__).resolve()
    if source!=out/'generate_dataset.py': shutil.copyfile(source,out/'generate_dataset.py')
    archive=package_dataset(out)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'seconds':round(time.time()-t,1),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),**stats},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
