import openpyxl
import csv
import xlwt

def csv2xls(csvname,xlsname):
	with open(csvname, 'r', encoding='utf-8') as f:
		read = csv.reader(f)
		workbook = xlwt.Workbook()
		sheet = workbook.add_sheet('sheet1')  # 创建一个sheet表格
		wb = openpyxl.Workbook()
		ws = wb.active
		l = 0
		for line in read:
			# print(line)
			r = 0
			for i in line:
				# print(i)
				sheet.write(l, r, i)  # 一个一个将单元格数据写入
				r = r + 1
			l = l + 1
		workbook.save(xlsname)  # 保存Excel
		print("xls saved")

if __name__=='__main__':
	csv2xls('网易财经.csv','企查查.xls')

