# Здесь содержатся возможные значения каждого из полей, передаваемых во входящем запросе

education_dict = ['Высшее - специалист', 'Неоконченное среднее', 'Среднее профессиональное', 'Среднее',
                  'Несколько высших', 'Бакалавр', 'Неоконченное высшее', 'MBA', 'Ученая степень']

employment_status_dict = ['Работаю по найму полный рабочий день/служу', 'Собственное дело',
                          'Работаю по найму неполный рабочий день', 'Студент', 'Пенсионер', 'Не работаю',
                          'Декретный отпуск']

family_status_dict = ['Никогда в браке не состоял(а)', 'Женат / замужем', 'Разведён / Разведена',
                      'Гражданский брак / совместное проживание', 'Вдовец / вдова']

merch_code_dict = ['77', '27', '19', '34', '75', '33', '8', '15', '80', '78', '43', '61', '18', '72', '45', '48', '67',
                   '12', '11', '40', '36', '31', '3', '47', '25', '30', '22', '70', '29', '6', '28', '24', '38', '63',
                   '44', '32', '10', '74', '50', '23', '49', '69', '4', '1', '39', '66', '13', '16', '2', '9', '14',
                   '21', '62', '64', '79', '71', '26', '17', '76', '73', '20', '46', '35', '7', '5', '41', '42', '37',
                   '68', '65']

goods_category_dict = ['Furniture', 'Fitness', 'Medical_services', 'Education', 'Other', 'Travel', 'Mobile_devices']

# Справочник содержащий названия полей входящего вектора, необходимого для корректной отработки модели
columns = ['Age', 'Seniority', 'DebtRatio', 'education_MBA', 'education_Бакалавр', 'education_Высшее - специалист',
           'education_Магистр', 'education_Неоконченное высшее', 'education_Неоконченное среднее',
           'education_Несколько высших', 'education_Среднее', 'education_Среднее профессиональное',
           'education_Ученая степень', 'employment status_Декретный отпуск', 'employment status_Не работаю',
           'employment status_Пенсионер', 'employment status_Работаю по найму неполный рабочий день',
           'employment status_Работаю по найму полный рабочий день/служу',
           'employment status_Собственное дело', 'employment status_Студент', 'Gender_Female', 'Gender_Male',
           'Family status_Вдовец / вдова', 'Family status_Гражданский брак / совместное проживание',
           'Family status_Женат / замужем', 'Family status_Никогда в браке не состоял(а)',
           'Family status_Разведён / Разведена', 'ChildCount_0.0', 'ChildCount_1.0', 'ChildCount_2.0', 'ChildCount_3.0',
           'ChildCount_4.0', 'ChildCount_5.0', 'SNILS_Empty', 'SNILS_Not_empty', 'Merch_code_1.0', 'Merch_code_10.0',
           'Merch_code_11.0', 'Merch_code_12.0', 'Merch_code_13.0', 'Merch_code_14.0', 'Merch_code_15.0',
           'Merch_code_16.0', 'Merch_code_17.0', 'Merch_code_18.0', 'Merch_code_19.0', 'Merch_code_2.0',
           'Merch_code_20.0', 'Merch_code_21.0', 'Merch_code_22.0', 'Merch_code_23.0', 'Merch_code_24.0',
           'Merch_code_25.0', 'Merch_code_26.0', 'Merch_code_27.0', 'Merch_code_28.0', 'Merch_code_29.0',
           'Merch_code_3.0', 'Merch_code_30.0', 'Merch_code_31.0', 'Merch_code_32.0', 'Merch_code_33.0',
           'Merch_code_34.0', 'Merch_code_35.0', 'Merch_code_36.0', 'Merch_code_37.0', 'Merch_code_38.0',
           'Merch_code_39.0', 'Merch_code_4.0', 'Merch_code_40.0', 'Merch_code_41.0', 'Merch_code_42.0',
           'Merch_code_43.0', 'Merch_code_44.0', 'Merch_code_45.0', 'Merch_code_46.0', 'Merch_code_47.0',
           'Merch_code_48.0', 'Merch_code_49.0', 'Merch_code_5.0', 'Merch_code_50.0', 'Merch_code_6.0',
           'Merch_code_61.0', 'Merch_code_62.0', 'Merch_code_63.0', 'Merch_code_64.0', 'Merch_code_65.0',
           'Merch_code_66.0', 'Merch_code_67.0', 'Merch_code_68.0', 'Merch_code_69.0', 'Merch_code_7.0',
           'Merch_code_70.0', 'Merch_code_71.0', 'Merch_code_72.0', 'Merch_code_73.0', 'Merch_code_74.0',
           'Merch_code_75.0', 'Merch_code_76.0', 'Merch_code_77.0', 'Merch_code_78.0', 'Merch_code_79.0',
           'Merch_code_8.0', 'Merch_code_80.0', 'Merch_code_9.0', 'Goods_category_Education', 'Goods_category_Fitness',
           'Goods_category_Furniture', 'Goods_category_Medical_services', 'Goods_category_Mobile_devices',
           'Goods_category_Other', 'Goods_category_Travel']
