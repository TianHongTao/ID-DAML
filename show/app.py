#-*-coding:utf-8 -*-
import tornado.ioloop
import tornado.web
import json

"""
写了个简单的JsonBaseAction的Handler 以json形式返回数据，实现了get和post两种常见api方式
估计本项目只用3个api，所以此处懒得分成多个py文件了，使用方式如下：
继承此BaseJsonHandler后
+ 对于get方法填充gen_get_action_data
+ 对于post方法填充gen_post_action_data
通用返回self.gen_data(data, msg="", status=0)即可
"""

class BaseJsonHandler(tornado.web.RequestHandler):
    """解决JS跨域请求问题"""
    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Content-type', 'application/json')
    
    def get(self):
        self.set_default_headers()
        try:
            data = self.check_gen_data_type(self.gen_get_action_data())
        except AssertionError as e:
            self.write(self.to_api_json(data="", msg=str(e), status=9999))
            return
        self.write(self.to_api_json(data["data"]))

    def post(self):
        self.set_default_headers()
        try:
            data = self.check_gen_data_type(self.gen_post_action_data())
        except AssertionError as e:
            self.write(self.to_api_json(data="", msg=str(e), status=9999))
            return
        self.write(self.to_api_json(data["data"], data["status"], data["msg"]))

    def gen_get_action_data(self):
        """待用户填充处理get方法将需要返回的数据返回"""
        return {}

    def gen_post_action_data(self):
        """待用户填充处理post方法将需要返回的数据返回"""
        return {}

    def to_api_json(self, data, status=0, msg=""):
        return json.dumps({"status":status, "msg":msg, "data":data}, ensure_ascii=False)
    
    def check_gen_data_type(self, data):
        if not isinstance(data, dict):
            assert False, "data type should be a dict"
        check_item = ["msg", "status", "data"]
        for key in check_item:
            if data.get(key) is None:
                print(key)
                print(data)
                assert False, "data format is not supported"
        return data
    
    def gen_data(self, data, msg="", status=0):
        return {
            "status": status,
            "msg": msg,
            "data": data
        }


class MainHandler(BaseJsonHandler):
    def gen_get_action_data(self):
        return self.gen_data("test app")

if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/", MainHandler),
    ])
    application.listen(2020)
    tornado.ioloop.IOLoop.current().start()