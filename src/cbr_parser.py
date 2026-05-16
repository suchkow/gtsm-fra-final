import xml.etree.ElementTree as ET
from datetime import date, datetime

import pandas as pd
import requests

 
class CBRParser:
    ENDPOINT = 'https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx'
    SOAP_NS = 'http://schemas.xmlsoap.org/soap/envelope/'
    CBR_NS = 'http://web.cbr.ru/'

    def _call(self, action: str, params: dict) -> ET.Element:
        inner = ''.join(f'<{k}>{v}</{k}>' for k, v in params.items())

        body = (
            "<?xml version='1.0' encoding='utf-8'?>"
            f"<soap:Envelope xmlns:soap='{self.SOAP_NS}'>"
            '<soap:Body>'
            f"<{action} xmlns='{self.CBR_NS}'>{inner}</{action}>"
            '</soap:Body>'
            '</soap:Envelope>'
        )

        resp = requests.post(
            self.ENDPOINT,
            data = body.encode('utf-8'),
            headers = {'Content-Type': 'text/xml; charset=utf-8', 'SOAPAction': f"'{self.CBR_NS}{action}'"},
            timeout = 60
        )
        resp.raise_for_status()
        return ET.fromstring(resp.content)


    @staticmethod
    def _local(tag: str) -> str:
        return tag.split('}')[-1] if '}' in tag else tag

    def _iter(self, root: ET.Element, local_tag: str):
        for el in root.iter():
            if self._local(el.tag) == local_tag:
                yield el

    def _text(self, el: ET.Element, local_tag: str) -> str | None:
        for child in el:
            if self._local(child.tag) == local_tag:
                return child.text
        return None


    @staticmethod
    def _date(s: str) -> date | None:
        if not s:
            return None
        
        s = s.strip()[:19]
        
        for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%d.%m.%Y'):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _float(s: str) -> float | None:
        if not s:
            return None
        try:
            return float(s.replace(',', '.'))
        except ValueError:
            return None

    @staticmethod
    def _fmt(d: str) -> str:
        return datetime.strptime(d, '%Y-%m-%d').strftime('%Y-%m-%dT00:00:00')

    def _build_df(self, rows: list[dict], sort_col: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(sort_col).reset_index(drop=True)
    

    def ruonia(self, from_date: str, to_date: str) -> pd.DataFrame:
        root = self._call('Ruonia', {'fromDate': self._fmt(from_date), 'ToDate': self._fmt(to_date)})
        rows = [
            {
                'date': self._date(self._text(el, 'DT')),
                'rate': self._float(self._text(el, 'ruo')),
                'volume_bln': self._float(self._text(el, 'vol')),
            }
            for el in self._iter(root, 'ro')
        ]
        return self._build_df(rows, 'date')

    def ruonia_sv(self, from_date: str, to_date: str) -> pd.DataFrame:
        root = self._call('RuoniaSV', {'fromDate': self._fmt(from_date), 'ToDate': self._fmt(to_date)})
        rows = [
            {
                'date':   self._date(self._text(el, 'DT')),
                'index':  self._float(self._text(el, 'RUONIA_Index')),
                '1M': self._float(self._text(el, 'RUONIA_AVG_1M')),
                '3M': self._float(self._text(el, 'RUONIA_AVG_3M')),
                '6M': self._float(self._text(el, 'RUONIA_AVG_6M')),
            }
            for el in self._iter(root, 'ra')
        ]
        return self._build_df(rows, 'date')

    def mkr(self, from_date: str, to_date: str, rate_type: int = 3) -> pd.DataFrame:
        '''
        Rate types:
        1 - MIBID (bid)
        2 - MIBOR (offer)
        3 - MIACR RUB - default
        4 - MIACR-IG RUB
        7 - MIACR-B RUB
        '''
        root = self._call('MKR', {'fromDate': self._fmt(from_date), 'ToDate': self._fmt(to_date)})
        rows = []
        for el in self._iter(root, 'MKR'):
            p1 = self._text(el, 'p1')
            if rate_type is not None and p1 != str(rate_type):
                continue
            row = {
                'date': self._date(self._text(el, 'CDate')),
                'type': int(p1) if p1 else None,
                'O/N': self._float(self._text(el, 'd1')),
                '1W': self._float(self._text(el, 'd7')),
                '1M': self._float(self._text(el, 'd30')),
                '3M': self._float(self._text(el, 'd90')),
                '6M': self._float(self._text(el, 'd180')),
                '1Y': self._float(self._text(el, 'd360')),
            }
            rows.append(row)
        df = self._build_df(rows, 'date')
        
        if rate_type is not None and not df.empty:
            df = df.drop(columns=['type'])
        return df

    def key_rate(self, from_date: str, to_date: str) -> pd.DataFrame:
        root = self._call('KeyRate', {'fromDate': self._fmt(from_date), 'ToDate': self._fmt(to_date)})
        rows = [
            {
                'date': self._date(self._text(el, 'DT')),
                'rate': self._float(self._text(el, 'Rate')),
            }
            for el in self._iter(root, 'KR')
        ]
        return self._build_df(rows, 'date')

    def key_rate_daily(self, from_date: str, to_date: str) -> pd.DataFrame:
        kr = self.key_rate(from_date, to_date).drop_duplicates('date').set_index('date')['rate']
        kr.index = pd.DatetimeIndex(kr.index)
        daily_idx = pd.date_range(from_date, to_date, freq='D')
        kr = kr.reindex(daily_idx, method='ffill').rename_axis('date').reset_index()
        kr['date'] = kr['date'].dt.date
        return kr

    def roisfix(self, from_date: str, to_date: str) -> pd.DataFrame:
        root = self._call('ROISfix', {'fromDate': self._fmt(from_date), 'ToDate': self._fmt(to_date)})
        rows = [
            {
                'date': self._date(self._text(el, 'D0')),
                '1W':   self._float(self._text(el, 'R1W')),
                '2W':   self._float(self._text(el, 'R2W')),
                '1M':   self._float(self._text(el, 'R1M')),
                '2M':   self._float(self._text(el, 'R2M')),
                '3M':   self._float(self._text(el, 'R3M')),
                '6M':   self._float(self._text(el, 'R6M')),
            }
            for el in self._iter(root, 'rf')
        ]
        return self._build_df(rows, 'date')